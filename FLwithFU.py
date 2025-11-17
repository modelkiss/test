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
import shutil
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


def resolve_base_dataset(subset):
    """递归解析出Subset链条中的最底层dataset"""
    base_dataset = subset
    while isinstance(base_dataset, torch.utils.data.Subset):
        base_dataset = base_dataset.dataset
    return base_dataset


def identify_forget_clients(client_raw_indices, train_targets, target_class):
    forget_clients = []
    for client_id, indices in enumerate(client_raw_indices):
        if any(train_targets[idx] == target_class for idx in indices):
            forget_clients.append(client_id)
    return forget_clients


def create_filtered_client_datasets(client_datasets, client_raw_indices, train_targets, target_class):
    base_dataset = resolve_base_dataset(client_datasets[0])
    filtered_datasets = []
    filtered_sizes = {}

    for client_id, original_dataset in enumerate(client_datasets):
        filtered_indices = filter_target_class(client_raw_indices[client_id], train_targets, target_class)
        if len(filtered_indices) == len(client_raw_indices[client_id]):
            # 没有目标类数据，直接复用原始子集
            filtered_datasets.append(original_dataset)
        else:
            filtered_subset = torch.utils.data.Subset(base_dataset, filtered_indices)
            filtered_datasets.append(filtered_subset)
            filtered_sizes[client_id] = len(filtered_indices)

    return filtered_datasets, filtered_sizes


def compute_client_data_sizes(client_raw_indices):
    return [len(indices) for indices in client_raw_indices]


def blend_model_parameters(original_params, updated_params, retain_weight, updated_weight):
    """根据样本权重融合两个模型参数字典"""
    if updated_weight == 0:
        return copy.deepcopy(original_params)

    total_weight = retain_weight + updated_weight
    if total_weight == 0:
        return copy.deepcopy(original_params)

    blended_params = {}
    for name in original_params.keys():
        orig_param = original_params[name].to(dtype=torch.float32)
        new_param = updated_params[name].to(dtype=torch.float32)
        blended = (retain_weight * orig_param + updated_weight * new_param) / total_weight
        blended_params[name] = blended.to(dtype=original_params[name].dtype)

    return blended_params


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
def client_train(client_model, client_dataset, epochs=5, batch_size=64, lr=0.001, trainable_param_names=None):
    client_model.train()
    client_model.to(device)

    dataset_size = len(client_dataset)
    if dataset_size == 0:
        # 没有样本时直接返回原始参数，避免除零
        return copy.deepcopy(client_model.state_dict()), 0

    if trainable_param_names is not None:
        trainable_param_names = set(trainable_param_names)
        for name, param in client_model.named_parameters():
            param.requires_grad = name in trainable_param_names
    else:
        for _, param in client_model.named_parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, client_model.parameters()), lr=lr)

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

        epoch_loss = running_loss / max(1, len(dataloader.dataset))

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


def evaluate_model(model, test_loader, return_class_acc=False, exclude_classes=None):
    """扩展：支持返回各类别准确率，并可排除指定类别"""
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    if exclude_classes is None:
        exclude_classes = set()
    else:
        exclude_classes = set(exclude_classes)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 跳过需要排除的类别
            mask = torch.tensor(
                [label.item() not in exclude_classes for label in labels], device=device, dtype=torch.bool
            )
            if mask.sum() == 0:
                continue

            inputs = inputs[mask]
            labels = labels[mask]
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

    if total == 0:
        overall_acc = 0.0
    else:
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
                       num_clients=10, client_epochs=5, batch_size=64, lr=0.001, target_class=None):
    """重训练方式：过滤数据后重新训练所有客户端并聚合"""
    # 1. 随机选择目标遗忘类别
    if target_class is None:
        target_class = np.random.randint(0, 10)
    target_class_name = CIFAR10_CLASSES[target_class]
    print(f"\n=== 重训练遗忘流程 ===")
    print(f"目标遗忘类别：{target_class} ({target_class_name})")

    # 2. 识别持有目标类数据的客户端（遗忘客户端）
    forget_client_ids = identify_forget_clients(client_raw_indices, train_targets, target_class)

    print(f"持有目标类数据的客户端：{forget_client_ids} (共{len(forget_client_ids)}个)")

    # 3. 准备遗忘后的客户端数据集
    new_client_datasets, filtered_sizes = create_filtered_client_datasets(
        client_datasets, client_raw_indices, train_targets, target_class
    )
    for client_id in forget_client_ids:
        original_size = len(client_raw_indices[client_id])
        filtered_size = filtered_sizes.get(client_id, original_size)
        print(f"客户端{client_id}：原始数据{original_size}个 → 过滤后{filtered_size}个（移除目标类数据）")

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
        if client_size > 0:
            client_params_list.append((client_params, client_size))

    # 5. 重新聚合得到遗忘后的全局模型
    print("\n重新聚合客户端模型参数...")
    if client_params_list:
        unlearned_global_params = fed_avg_aggregate(client_params_list)
    else:
        unlearned_global_params = copy.deepcopy(global_model.state_dict())
    unlearned_global_model = CIFAR10CNN().to(device)
    unlearned_global_model.load_state_dict(unlearned_global_params)

    # 6. 评估遗忘前后性能
    return unlearned_global_model, target_class, evaluate_performance(global_model, unlearned_global_model, test_loader)


def fru_unlearning(global_model, client_datasets, client_raw_indices, train_targets, test_loader,
                   num_clients=10, client_epochs=5, batch_size=64, lr=0.001, target_class=None):
    """FRU (Federated Retrieval Unlearning) 方法"""
    print("\n=== FRU遗忘流程 ===")
    if target_class is None:
        target_class = np.random.randint(0, 10)
    print(f"目标遗忘类别：{target_class} ({CIFAR10_CLASSES[target_class]})")

    forget_client_ids = identify_forget_clients(client_raw_indices, train_targets, target_class)
    print(f"持有目标类数据的客户端：{forget_client_ids} (共{len(forget_client_ids)}个)")

    if not forget_client_ids:
        print("没有客户端持有目标类别数据，模型保持不变")
        return copy.deepcopy(global_model), target_class, evaluate_performance(global_model, global_model, test_loader)

    filtered_datasets, _ = create_filtered_client_datasets(
        client_datasets, client_raw_indices, train_targets, target_class
    )

    client_original_sizes = compute_client_data_sizes(client_raw_indices)
    total_samples = sum(client_original_sizes)
    retained_weight = total_samples - sum(client_original_sizes[c] for c in forget_client_ids)

    forget_params_list = []
    updated_weight = 0
    for client_id in forget_client_ids:
        filtered_dataset = filtered_datasets[client_id]
        filtered_size = len(filtered_dataset)
        if filtered_size == 0:
            print(f"客户端{client_id} 过滤后无样本，跳过训练")
            continue

        print(f"客户端{client_id}：过滤后样本数 {filtered_size}")
        client_model = CIFAR10CNN().to(device)
        client_model.load_state_dict(global_model.state_dict())
        client_params, client_size = client_train(
            client_model, filtered_dataset,
            epochs=max(1, client_epochs // 2), batch_size=batch_size, lr=lr
        )
        if client_size > 0:
            forget_params_list.append((client_params, client_size))
            updated_weight += client_size

    if not forget_params_list:
        print("遗忘客户端在过滤后没有可用样本，模型保持原样")
        return copy.deepcopy(global_model), target_class, evaluate_performance(global_model, global_model, test_loader)

    forget_aggregated_params = fed_avg_aggregate(forget_params_list)
    blended_params = blend_model_parameters(
        global_model.state_dict(), forget_aggregated_params,
        retain_weight=retained_weight, updated_weight=updated_weight
    )

    unlearned_model = CIFAR10CNN().to(device)
    unlearned_model.load_state_dict(blended_params)

    return unlearned_model, target_class, evaluate_performance(global_model, unlearned_model, test_loader)


def fed_eraser_unlearning(global_model, client_datasets, client_raw_indices, train_targets, test_loader,
                          num_clients=10, client_epochs=5, batch_size=64, lr=0.001, target_class=None):
    """FedEraser 方法"""
    print("\n=== FedEraser遗忘流程 ===")
    if target_class is None:
        target_class = np.random.randint(0, 10)
    print(f"目标遗忘类别：{target_class} ({CIFAR10_CLASSES[target_class]})")

    forget_client_ids = identify_forget_clients(client_raw_indices, train_targets, target_class)
    print(f"移除贡献的客户端：{forget_client_ids} (共{len(forget_client_ids)}个)")

    if len(forget_client_ids) == num_clients:
        print("所有客户端均需遗忘，退化为重训练策略")
        return retrain_unlearning(
            global_model, client_datasets, client_raw_indices, train_targets, test_loader,
            num_clients=num_clients, client_epochs=client_epochs, batch_size=batch_size, lr=lr,
            target_class=target_class
        )

    retained_client_ids = [cid for cid in range(num_clients) if cid not in forget_client_ids]
    client_params_list = []
    for client_id in retained_client_ids:
        dataset = client_datasets[client_id]
        if len(dataset) == 0:
            continue

        client_model = CIFAR10CNN().to(device)
        client_model.load_state_dict(global_model.state_dict())
        client_params, client_size = client_train(
            client_model, dataset,
            epochs=client_epochs, batch_size=batch_size, lr=lr
        )
        if client_size > 0:
            client_params_list.append((client_params, client_size))

    if not client_params_list:
        print("没有可用于恢复的客户端，模型保持原样")
        return copy.deepcopy(global_model), target_class, evaluate_performance(global_model, global_model, test_loader)

    new_params = fed_avg_aggregate(client_params_list)
    unlearned_model = CIFAR10CNN().to(device)
    unlearned_model.load_state_dict(new_params)

    return unlearned_model, target_class, evaluate_performance(global_model, unlearned_model, test_loader)


def fed_recovery_unlearning(global_model, client_datasets, client_raw_indices, train_targets, test_loader,
                            num_clients=10, client_epochs=5, batch_size=64, lr=0.001, target_class=None):
    """FedRecovery 方法"""
    print("\n=== FedRecovery遗忘流程 ===")
    if target_class is None:
        target_class = np.random.randint(0, 10)
    print(f"目标遗忘类别：{target_class} ({CIFAR10_CLASSES[target_class]})")

    forget_client_ids = identify_forget_clients(client_raw_indices, train_targets, target_class)
    print(f"需要恢复影响的客户端：{forget_client_ids} (共{len(forget_client_ids)}个)")

    # 第一步：执行FedEraser逻辑，移除目标客户端的贡献
    erased_model, _, _ = fed_eraser_unlearning(
        global_model, client_datasets, client_raw_indices, train_targets, test_loader,
        num_clients=num_clients, client_epochs=max(1, client_epochs // 2),
        batch_size=batch_size, lr=lr, target_class=target_class
    )

    # 第二步：选择部分未受影响客户端进行恢复性训练
    retained_client_ids = [cid for cid in range(num_clients) if cid not in forget_client_ids]
    recovery_client_count = min(len(retained_client_ids), max(1, num_clients // 3))
    recovery_clients = retained_client_ids[:recovery_client_count]
    print(f"恢复阶段选取的客户端：{recovery_clients}")

    recovery_params_list = []
    for client_id in recovery_clients:
        dataset = client_datasets[client_id]
        if len(dataset) == 0:
            continue

        client_model = CIFAR10CNN().to(device)
        client_model.load_state_dict(erased_model.state_dict())
        client_params, client_size = client_train(
            client_model, dataset,
            epochs=max(1, client_epochs // 2), batch_size=batch_size, lr=lr
        )
        if client_size > 0:
            recovery_params_list.append((client_params, client_size))

    if recovery_params_list:
        recovery_params = fed_avg_aggregate(recovery_params_list)
        erased_model.load_state_dict(recovery_params)
    else:
        print("恢复阶段没有可用客户端，跳过额外聚合")

    return erased_model, target_class, evaluate_performance(global_model, erased_model, test_loader)


def sfu_unlearning(global_model, client_datasets, client_raw_indices, train_targets, test_loader,
                   num_clients=10, client_epochs=5, batch_size=64, lr=0.001, target_class=None):
    """SFU (Selective Federated Unlearning) 方法"""
    print("\n=== SFU遗忘流程 ===")
    if target_class is None:
        target_class = np.random.randint(0, 10)
    print(f"目标遗忘类别：{target_class} ({CIFAR10_CLASSES[target_class]})")

    forget_client_ids = identify_forget_clients(client_raw_indices, train_targets, target_class)
    print(f"选择性微调的客户端：{forget_client_ids} (共{len(forget_client_ids)}个)")

    if not forget_client_ids:
        print("没有客户端需要遗忘，模型保持原样")
        return copy.deepcopy(global_model), target_class, evaluate_performance(global_model, global_model, test_loader)

    filtered_datasets, _ = create_filtered_client_datasets(
        client_datasets, client_raw_indices, train_targets, target_class
    )

    trainable_names = [name for name, _ in CIFAR10CNN().named_parameters() if name.startswith('fc_layers')]

    client_original_sizes = compute_client_data_sizes(client_raw_indices)
    total_samples = sum(client_original_sizes)
    retained_weight = total_samples - sum(client_original_sizes[c] for c in forget_client_ids)

    updated_weight = 0
    forget_params_list = []
    for client_id in forget_client_ids:
        dataset = filtered_datasets[client_id]
        if len(dataset) == 0:
            print(f"客户端{client_id} 过滤后无样本，跳过")
            continue

        client_model = CIFAR10CNN().to(device)
        client_model.load_state_dict(global_model.state_dict())
        params, size = client_train(
            client_model, dataset,
            epochs=max(1, client_epochs // 2), batch_size=batch_size, lr=lr,
            trainable_param_names=trainable_names
        )
        if size > 0:
            forget_params_list.append((params, size))
            updated_weight += size

    if not forget_params_list:
        print("选择性训练未获得有效更新，模型保持原样")
        return copy.deepcopy(global_model), target_class, evaluate_performance(global_model, global_model, test_loader)

    aggregated_params = fed_avg_aggregate(forget_params_list)
    blended_params = blend_model_parameters(
        global_model.state_dict(), aggregated_params,
        retain_weight=retained_weight, updated_weight=updated_weight
    )

    unlearned_model = CIFAR10CNN().to(device)
    unlearned_model.load_state_dict(blended_params)

    return unlearned_model, target_class, evaluate_performance(global_model, unlearned_model, test_loader)


def fed_af_unlearning(global_model, client_datasets, client_raw_indices, train_targets, test_loader,
                      num_clients=10, client_epochs=5, batch_size=64, lr=0.001, target_class=None):
    """FedAF (Federated Amnesic Fine-tuning) 方法"""
    print("\n=== FedAF遗忘流程 ===")
    if target_class is None:
        target_class = np.random.randint(0, 10)
    print(f"目标遗忘类别：{target_class} ({CIFAR10_CLASSES[target_class]})")

    forget_client_ids = identify_forget_clients(client_raw_indices, train_targets, target_class)
    print(f"进行失忆微调的客户端：{forget_client_ids} (共{len(forget_client_ids)}个)")

    base_dataset = resolve_base_dataset(client_datasets[0])
    forget_indices = [
        idx
        for client_id in forget_client_ids
        for idx in client_raw_indices[client_id]
        if train_targets[idx] == target_class
    ]

    if not forget_indices:
        print("目标类别样本为空，模型保持原样")
        return copy.deepcopy(global_model), target_class, evaluate_performance(global_model, global_model, test_loader)

    forget_subset = torch.utils.data.Subset(base_dataset, forget_indices)
    dataloader = torch.utils.data.DataLoader(
        forget_subset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    unlearned_model = CIFAR10CNN().to(device)
    unlearned_model.load_state_dict(global_model.state_dict())
    unlearned_model.train()
    optimizer = optim.Adam(unlearned_model.parameters(), lr=lr)
    epochs = max(1, client_epochs // 2)

    print("对目标类别执行失忆微调，使其输出接近均匀分布...")
    for epoch in range(epochs):
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = unlearned_model(inputs)
            log_probs = torch.log_softmax(outputs, dim=1)
            target_dist = torch.full((inputs.size(0), 10), 1.0 / 10, device=device)
            loss = torch.nn.functional.kl_div(log_probs, target_dist, reduction='batchmean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    unlearned_model.eval()

    # 使用一小部分未受影响客户端进行恢复性微调
    retained_client_ids = [cid for cid in range(num_clients) if cid not in forget_client_ids]
    support_client_count = min(len(retained_client_ids), max(1, num_clients // 4))
    support_clients = retained_client_ids[:support_client_count]
    print(f"支持性微调客户端：{support_clients}")

    support_params_list = []
    for client_id in support_clients:
        dataset = client_datasets[client_id]
        if len(dataset) == 0:
            continue

        client_model = CIFAR10CNN().to(device)
        client_model.load_state_dict(unlearned_model.state_dict())
        params, size = client_train(
            client_model, dataset,
            epochs=1, batch_size=batch_size, lr=lr
        )
        if size > 0:
            support_params_list.append((params, size))

    if support_params_list:
        support_params = fed_avg_aggregate(support_params_list)
        unlearned_model.load_state_dict(support_params)

    return unlearned_model, target_class, evaluate_performance(global_model, unlearned_model, test_loader)


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

    class ModelSaveManager:
        def __init__(self, base_dir):
            self.base_dir = base_dir
            self.temp_dir = os.path.join(base_dir, "temp")
            os.makedirs(self.temp_dir, exist_ok=True)
            self.stage_paths = defaultdict(list)

        def save_temp_model(self, stage, label, state_dict):
            stage_dir = os.path.join(self.temp_dir, stage)
            os.makedirs(stage_dir, exist_ok=True)
            filename = f"{stage}_{label}.pth"
            path = os.path.join(stage_dir, filename)
            torch.save(state_dict, path)
            self.stage_paths[stage].append(path)
            return path

        def finalize(self, stage, keep_last_n, target_dir):
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir, exist_ok=True)

            kept_paths = self.stage_paths.get(stage, [])[-keep_last_n:]
            for idx, path in enumerate(kept_paths, start=1):
                dest = os.path.join(target_dir, f"{stage}_aggregation_{idx}.pth")
                shutil.copy(path, dest)

            # 清理临时文件
            for stage_path in self.stage_paths.get(stage, []):
                if os.path.exists(stage_path) and stage_path not in kept_paths:
                    os.remove(stage_path)
            stage_dir = os.path.join(self.temp_dir, stage)
            if os.path.exists(stage_dir) and not os.listdir(stage_dir):
                os.rmdir(stage_dir)

    def create_excluded_test_loader(test_dataset, excluded_class, batch_size=100):
        filtered_indices = [idx for idx, label in enumerate(test_dataset.targets) if label != excluded_class]
        filtered_subset = torch.utils.data.Subset(test_dataset, filtered_indices)
        return torch.utils.data.DataLoader(filtered_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 超参数设置
    num_clients = 10
    client_epochs = 5
    batch_size = 64
    lr = 0.001
    non_iid_alpha = 0.5
    client_fraction = 0.5
    target_accuracy = 95.0

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
    save_manager = ModelSaveManager(MODEL_SAVE_DIR)
    pre_round = 0

    while True:
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

        pre_round += 1
        # 评估
        accuracy = evaluate_model(global_model, test_loader)
        round_accuracies.append(accuracy)
        round_time = time.time() - start_time

        save_manager.save_temp_model("pre", f"round_{pre_round}", global_model.state_dict())
        print(f"第 {pre_round} 次聚合，测试准确率: {accuracy:.2f}%, 耗时: {round_time:.2f}s")

        if accuracy >= target_accuracy:
            print(f"达到预设准确率 {target_accuracy}%，开始执行遗忘流程")
            break

    # 3. 选择并执行联邦遗忘流程
    method_name, unlearning_func = select_unlearning_method()
    print(f"您选择的遗忘方式：{method_name}")
    unlearned_model, target_class, performance = unlearning_func(
        global_model, client_datasets, client_raw_indices, train_targets, test_loader,
        num_clients=num_clients, client_epochs=client_epochs, batch_size=batch_size, lr=lr
    )

    # 保存遗忘流程产生的模型视为遗忘后聚合的一部分
    save_manager.save_temp_model("post", "unlearning_result", unlearned_model.state_dict())

    # 使用过滤后的数据继续训练直至达到目标准确率
    print("\n=== 遗忘后继续联邦训练，直至达到目标准确率 ===")
    filtered_client_datasets, _ = create_filtered_client_datasets(
        client_datasets, client_raw_indices, train_targets, target_class
    )
    filtered_test_loader = create_excluded_test_loader(test_dataset, target_class, batch_size=100)

    post_round = 0
    post_accuracies = []
    current_params = unlearned_model.state_dict()

    while True:
        start_time = time.time()
        num_selected = max(1, int(num_clients * client_fraction))
        selected_clients = np.random.choice(range(num_clients), size=num_selected, replace=False)

        client_params_list = []
        for client_id in selected_clients:
            client_model = CIFAR10CNN().to(device)
            client_model.load_state_dict(current_params)
            client_params, client_size = client_train(
                client_model, filtered_client_datasets[client_id],
                epochs=client_epochs, batch_size=batch_size, lr=lr
            )
            client_params_list.append((client_params, client_size))

        current_params = fed_avg_aggregate(client_params_list)
        unlearned_model.load_state_dict(current_params)

        post_round += 1
        accuracy = evaluate_model(unlearned_model, filtered_test_loader, exclude_classes={target_class})
        post_accuracies.append(accuracy)
        round_time = time.time() - start_time

        save_manager.save_temp_model("post", f"round_{post_round}", unlearned_model.state_dict())
        print(f"遗忘后第 {post_round} 次聚合，过滤后测试准确率: {accuracy:.2f}%, 耗时: {round_time:.2f}s")

        if accuracy >= target_accuracy:
            print(f"遗忘后准确率达到 {target_accuracy}%，结束训练")
            break

    # 4. 结果可视化
    print("\n=== 结果可视化 ===")
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, pre_round + 1), round_accuracies, marker='o', linewidth=2, color='blue')
    plt.xlabel('联邦训练轮数（遗忘前）')
    plt.ylabel('测试准确率 (%)')
    plt.title('遗忘前联邦训练准确率变化')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(1, post_round + 1), post_accuracies, marker='o', linewidth=2, color='green')
    plt.xlabel('联邦训练轮数（遗忘后）')
    plt.ylabel('过滤后测试准确率 (%)')
    plt.title('遗忘后联邦训练准确率变化')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    classes = CIFAR10_CLASSES
    x = np.arange(len(classes))
    width = 0.35

    pre_acc = performance["pre_unlearn_class_acc"]
    post_acc = performance["post_unlearn_class_acc"]

    bars1 = plt.bar(x - width / 2, pre_acc, width, label='遗忘前', color='lightblue')
    bars2 = plt.bar(x + width / 2, post_acc, width, label='遗忘后', color='lightcoral')

    bars1[target_class].set_color('blue')
    bars2[target_class].set_color('red')

    plt.xlabel('类别')
    plt.ylabel('准确率 (%)')
    plt.title(f'遗忘前后各类别准确率对比（红色=遗忘类别：{classes[target_class]}）')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 整理需要保留的模型
    pre_save_dir = os.path.join(MODEL_SAVE_DIR, "pre_unlearning")
    post_save_dir = os.path.join(MODEL_SAVE_DIR, "post_unlearning")

    save_manager.finalize("pre", 5, pre_save_dir)
    save_manager.finalize("post", 5, post_save_dir)

    print("\n模型整理完成（仅保留前后各5个聚合模型）：")
    print(f"  - 遗忘前模型目录：{pre_save_dir}")
    print(f"  - 遗忘后模型目录：{post_save_dir}")


if __name__ == "__main__":
    main()
