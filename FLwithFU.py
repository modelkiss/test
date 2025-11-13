import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os
from collections import defaultdict

# 设备配置（全局只定义一次，避免重复打印）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR10类别名称（用于可视化）
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


# --------------------------
# 1. 数据准备与客户端划分
# --------------------------
def load_cifar10():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


def split_data_to_clients(train_dataset, num_clients=10, non_iid_alpha=0.5):
    targets = np.array(train_dataset.targets)
    num_classes = len(np.unique(targets))
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        dirichlet_dist = np.random.dirichlet([non_iid_alpha] * num_clients)
        class_size = len(class_indices[c])
        client_sizes = (dirichlet_dist * class_size).astype(int)
        client_sizes[:class_size % num_clients] += 1

        start = 0
        for i in range(num_clients):
            end = start + client_sizes[i]
            client_indices[i].extend(class_indices[c][start:end])
            start = end

    # 保存每个客户端的原始索引（用于后续遗忘时过滤）
    client_datasets = []
    client_raw_indices = []  # 记录每个客户端的原始数据索引
    for indices in client_indices:
        subset = torch.utils.data.Subset(train_dataset, indices)
        client_datasets.append(subset)
        client_raw_indices.append(indices)

    return client_datasets, client_raw_indices, train_dataset.targets


def filter_target_class(client_raw_indices, train_targets, target_class):
    """过滤客户端数据中的目标遗忘类别，返回过滤后的索引"""
    filtered_indices = []
    for idx in client_raw_indices:
        # 只保留非目标类的索引
        if train_targets[idx] != target_class:
            filtered_indices.append(idx)
    return filtered_indices


# --------------------------
# 2. 模型定义
# --------------------------
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# --------------------------
# 3. 联邦学习核心函数
# --------------------------
def client_train(client_model, client_dataset, epochs=5, batch_size=64, lr=0.001):
    client_model.train()
    client_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(client_model.parameters(), lr=lr)

    dataloader = torch.utils.data.DataLoader(
        client_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = client_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)

    return client_model.state_dict(), len(client_dataset)


def fed_avg_aggregate(client_params_list):
    """修复：显式指定浮点类型，解决类型不匹配问题"""
    total_size = sum(size for _, size in client_params_list)
    global_params = {}

    # 遍历所有参数名称，初始化全局参数（显式转换为float32）
    for param_name in client_params_list[0][0].keys():
        template_param = client_params_list[0][0][param_name]
        global_params[param_name] = torch.zeros_like(template_param, dtype=torch.float32)

    # 加权平均（确保所有参数都是float32类型）
    for client_params, client_size in client_params_list:
        weight = client_size / total_size
        for param_name in global_params.keys():
            client_param_float = client_params[param_name].to(dtype=torch.float32)
            global_params[param_name] += weight * client_param_float

    # 恢复原始参数类型
    ref_client_params = client_params_list[0][0]
    for param_name in global_params.keys():
        original_dtype = ref_client_params[param_name].dtype
        if original_dtype in (torch.int64, torch.long):
            global_params[param_name] = global_params[param_name].round().to(dtype=original_dtype)
        else:
            global_params[param_name] = global_params[param_name].to(dtype=original_dtype)

    return global_params


def evaluate_model(model, test_loader, return_class_acc=False):
    """扩展：支持返回各类别准确率"""
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 统计各类别准确率
            for label, pred in zip(labels, predicted):
                c = label.item()
                class_total[c] += 1
                if pred == label:
                    class_correct[c] += 1

    overall_acc = 100 * correct / total

    if not return_class_acc:
        return overall_acc

    # 计算各类别准确率
    class_acc = []
    for c in range(10):
        if class_total[c] == 0:
            class_acc.append(0.0)
        else:
            class_acc.append(100 * class_correct[c] / class_total[c])

    return overall_acc, class_acc


# --------------------------
# 4. 联邦遗忘方法实现
# --------------------------
def retrain_unlearning(global_model, client_datasets, client_raw_indices, train_targets, test_loader,
                       num_clients=10, client_epochs=5, batch_size=64, lr=0.001):
    """重训练方式：过滤数据后重新训练所有客户端并聚合"""
    # 1. 随机选择目标遗忘类别
    target_class = np.random.randint(0, 10)
    target_class_name = CIFAR10_CLASSES[target_class]
    print(f"\n=== 重训练遗忘流程 ===")
    print(f"目标遗忘类别：{target_class} ({target_class_name})")

    # 2. 识别持有目标类数据的客户端（遗忘客户端）
    forget_client_ids = []
    for client_id in range(num_clients):
        client_indices = client_raw_indices[client_id]
        has_target_class = any(train_targets[idx] == target_class for idx in client_indices)
        if has_target_class:
            forget_client_ids.append(client_id)

    print(f"持有目标类数据的客户端：{forget_client_ids} (共{len(forget_client_ids)}个)")

    # 3. 准备遗忘后的客户端数据集
    new_client_datasets = []
    for client_id in range(num_clients):
        if client_id in forget_client_ids:
            # 遗忘客户端：过滤目标类数据
            filtered_indices = filter_target_class(
                client_raw_indices[client_id], train_targets, target_class
            )
            print(f"客户端{client_id}：原始数据{len(client_raw_indices[client_id])}个 "
                  f"→ 过滤后{len(filtered_indices)}个（移除目标类数据）")
            new_subset = torch.utils.data.Subset(client_datasets[0].dataset, filtered_indices)
            new_client_datasets.append(new_subset)
        else:
            # 非遗忘客户端：保留原始数据
            new_client_datasets.append(client_datasets[client_id])

    # 4. 所有客户端重新训练
    print("\n所有客户端重新训练（遗忘客户端已过滤目标类数据）...")
    client_params_list = []
    for client_id in range(num_clients):
        client_model = CIFAR10CNN().to(device)
        client_model.load_state_dict(global_model.state_dict())

        client_params, client_size = client_train(
            client_model, new_client_datasets[client_id],
            epochs=client_epochs, batch_size=batch_size, lr=lr
        )
        client_params_list.append((client_params, client_size))

    # 5. 重新聚合得到遗忘后的全局模型
    print("\n重新聚合客户端模型参数...")
    unlearned_global_params = fed_avg_aggregate(client_params_list)
    unlearned_global_model = CIFAR10CNN().to(device)
    unlearned_global_model.load_state_dict(unlearned_global_params)

    # 6. 评估遗忘前后性能
    return unlearned_global_model, target_class, evaluate_performance(global_model, unlearned_global_model, test_loader)


def fru_unlearning(global_model, client_datasets, client_raw_indices, train_targets, test_loader,
                   num_clients=10, client_epochs=5, batch_size=64, lr=0.001):
    """FRU (Federated Retrieval Unlearning) 方法"""
    print("\n=== FRU遗忘流程 ===")
    print("FRU: 通过检索相关参数进行选择性遗忘（待实现）")
    # 这里添加FRU算法的具体实现逻辑
    target_class = np.random.randint(0, 10)
    # 临时返回原始模型作为占位
    return copy.deepcopy(global_model), target_class, evaluate_performance(global_model, global_model, test_loader)


def fed_eraser_unlearning(global_model, client_datasets, client_raw_indices, train_targets, test_loader,
                          num_clients=10, client_epochs=5, batch_size=64, lr=0.001):
    """FedEraser 方法"""
    print("\n=== FedEraser遗忘流程 ===")
    print("FedEraser: 通过参数擦除实现遗忘（待实现）")
    # 这里添加FedEraser算法的具体实现逻辑
    target_class = np.random.randint(0, 10)
    return copy.deepcopy(global_model), target_class, evaluate_performance(global_model, global_model, test_loader)


def fed_recovery_unlearning(global_model, client_datasets, client_raw_indices, train_targets, test_loader,
                            num_clients=10, client_epochs=5, batch_size=64, lr=0.001):
    """FedRecovery 方法"""
    print("\n=== FedRecovery遗忘流程 ===")
    print("FedRecovery: 通过模型恢复实现遗忘（待实现）")
    # 这里添加FedRecovery算法的具体实现逻辑
    target_class = np.random.randint(0, 10)
    return copy.deepcopy(global_model), target_class, evaluate_performance(global_model, global_model, test_loader)


def sfu_unlearning(global_model, client_datasets, client_raw_indices, train_targets, test_loader,
                   num_clients=10, client_epochs=5, batch_size=64, lr=0.001):
    """SFU (Selective Federated Unlearning) 方法"""
    print("\n=== SFU遗忘流程 ===")
    print("SFU: 选择性联邦遗忘（待实现）")
    # 这里添加SFU算法的具体实现逻辑
    target_class = np.random.randint(0, 10)
    return copy.deepcopy(global_model), target_class, evaluate_performance(global_model, global_model, test_loader)


def fed_af_unlearning(global_model, client_datasets, client_raw_indices, train_targets, test_loader,
                      num_clients=10, client_epochs=5, batch_size=64, lr=0.001):
    """FedAF (Federated Amnesic Fine-tuning) 方法"""
    print("\n=== FedAF遗忘流程 ===")
    print("FedAF: 联邦失忆微调（待实现）")
    # 这里添加FedAF算法的具体实现逻辑
    target_class = np.random.randint(0, 10)
    return copy.deepcopy(global_model), target_class, evaluate_performance(global_model, global_model, test_loader)


def evaluate_performance(pre_model, post_model, test_loader):
    """评估遗忘前后的性能差异"""
    pre_acc, pre_class_acc = evaluate_model(pre_model, test_loader, return_class_acc=True)
    post_acc, post_class_acc = evaluate_model(post_model, test_loader, return_class_acc=True)

    print("\n=== 遗忘前后性能对比 ===")
    print(f"遗忘前整体准确率：{pre_acc:.2f}%")
    print(f"遗忘后整体准确率：{post_acc:.2f}%")
    print(f"\n遗忘前各类别准确率：")
    for c in range(10):
        print(f"  {CIFAR10_CLASSES[c]}: {pre_class_acc[c]:.2f}%")
    print(f"\n遗忘后各类别准确率：")
    for c in range(10):
        print(f"  {CIFAR10_CLASSES[c]}: {post_class_acc[c]:.2f}%")

    return {
        "pre_unlearn_acc": pre_acc,
        "post_unlearn_acc": post_acc,
        "pre_unlearn_class_acc": pre_class_acc,
        "post_unlearn_class_acc": post_class_acc
    }


def select_unlearning_method():
    """让用户选择遗忘方式"""
    print("\n请选择联邦遗忘方式：")
    print("1. 重训练 (Retrain)")
    print("2. FRU (Federated Retrieval Unlearning)")
    print("3. FedEraser")
    print("4. FedRecovery")
    print("5. SFU (Selective Federated Unlearning)")
    print("6. FedAF (Federated Amnesic Fine-tuning)")

    while True:
        try:
            choice = int(input("请输入选项编号 (1-6): "))
            if 1 <= choice <= 6:
                return {
                    1: ("重训练", retrain_unlearning),
                    2: ("FRU", fru_unlearning),
                    3: ("FedEraser", fed_eraser_unlearning),
                    4: ("FedRecovery", fed_recovery_unlearning),
                    5: ("SFU", sfu_unlearning),
                    6: ("FedAF", fed_af_unlearning)
                }[choice]
            else:
                print("请输入1-6之间的数字")
        except ValueError:
            print("请输入有效的数字")


# --------------------------
# 5. 主流程
# --------------------------
def main():
    # 初始化操作
    print(f"使用设备: {device}")

    # 模型保存文件夹配置
    MODEL_SAVE_DIR = "./model"
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"创建模型保存文件夹：{MODEL_SAVE_DIR}")
    else:
        print(f"模型保存文件夹已存在：{MODEL_SAVE_DIR}")

    # 超参数设置
    num_clients = 10
    num_rounds = 20
    client_epochs = 5
    batch_size = 64
    lr = 0.001
    non_iid_alpha = 0.5
    client_fraction = 0.5

    # 1. 加载数据并划分客户端
    print("\n加载CIFAR10数据集...")
    train_dataset, test_dataset = load_cifar10()
    print(f"将数据划分给 {num_clients} 个客户端（Non-IID分布）...")
    client_datasets, client_raw_indices, train_targets = split_data_to_clients(
        train_dataset, num_clients=num_clients, non_iid_alpha=non_iid_alpha
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=2
    )

    # 2. 正常联邦训练（得到预训练全局模型）
    print("\n=== 开始正常联邦训练 ===")
    global_model = CIFAR10CNN().to(device)
    global_params = global_model.state_dict()
    round_accuracies = []

    for round_idx in range(num_rounds):
        start_time = time.time()
        num_selected = max(1, int(num_clients * client_fraction))
        selected_clients = np.random.choice(range(num_clients), size=num_selected, replace=False)

        client_params_list = []
        for client_id in selected_clients:
            client_model = CIFAR10CNN().to(device)
            client_model.load_state_dict(global_params)
            client_params, client_size = client_train(
                client_model, client_datasets[client_id],
                epochs=client_epochs, batch_size=batch_size, lr=lr
            )
            client_params_list.append((client_params, client_size))

        # 聚合模型
        global_params = fed_avg_aggregate(client_params_list)
        global_model.load_state_dict(global_params)

        # 评估
        accuracy = evaluate_model(global_model, test_loader)
        round_accuracies.append(accuracy)
        round_time = time.time() - start_time

        print(f"第 {round_idx + 1}/{num_rounds} 轮，测试准确率: {accuracy:.2f}%, 耗时: {round_time:.2f}s")

    # 3. 选择并执行联邦遗忘流程
    method_name, unlearning_func = select_unlearning_method()
    print(f"您选择的遗忘方式：{method_name}")
    unlearned_model, target_class, performance = unlearning_func(
        global_model, client_datasets, client_raw_indices, train_targets, test_loader,
        num_clients=num_clients, client_epochs=client_epochs, batch_size=batch_size, lr=lr
    )

    # 4. 结果可视化
    print("\n=== 结果可视化 ===")
    # 图1：正常联邦训练的准确率曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_rounds + 1), round_accuracies, marker='o', linewidth=2, color='blue')
    plt.xlabel('联邦训练轮数')
    plt.ylabel('测试准确率 (%)')
    plt.title('正常联邦训练准确率变化')
    plt.grid(True)

    # 图2：遗忘前后各类别准确率对比
    plt.subplot(1, 2, 2)
    classes = CIFAR10_CLASSES
    x = np.arange(len(classes))
    width = 0.35

    pre_acc = performance["pre_unlearn_class_acc"]
    post_acc = performance["post_unlearn_class_acc"]

    bars1 = plt.bar(x - width / 2, pre_acc, width, label='遗忘前', color='lightblue')
    bars2 = plt.bar(x + width / 2, post_acc, width, label='遗忘后', color='lightcoral')

    # 高亮目标类别
    bars1[target_class].set_color('blue')
    bars2[target_class].set_color('red')

    plt.xlabel('类别')
    plt.ylabel('准确率 (%)')
    plt.title(f'遗忘前后各类别准确率对比（红色=遗忘类别：{classes[target_class]}）')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 保存模型
    pre_unlearn_model_path = os.path.join(MODEL_SAVE_DIR, f"federated_cifar10_pre_{method_name}.pth")
    post_unlearn_model_path = os.path.join(MODEL_SAVE_DIR, f"federated_cifar10_post_{method_name}.pth")

    torch.save(global_model.state_dict(), pre_unlearn_model_path)
    torch.save(unlearned_model.state_dict(), post_unlearn_model_path)

    print("\n模型保存完成：")
    print(f"  - 遗忘前模型：{pre_unlearn_model_path}")
    print(f"  - 遗忘后模型：{post_unlearn_model_path}")


if __name__ == "__main__":
    main()