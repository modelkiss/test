import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# 导入FLwithFU中的模型和数据处理函数
from FLwithFU import CIFAR10CNN, CIFAR10_CLASSES

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据加载与预处理（保持与原代码一致）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=transform
)

subset_ratio = 0.1
if hasattr(dataset, "targets"):
    targets = np.array(dataset.targets)
elif hasattr(dataset, "train_labels"):
    targets = np.array(dataset.train_labels)
else:
    targets = np.array(dataset.test_labels)

num_classes = len(getattr(dataset, "classes", np.unique(targets)))
subset_indices = []

for class_label in range(num_classes):
    class_indices = np.where(targets == class_label)[0]
    num_samples = max(1, int(len(class_indices) * subset_ratio))
    selected_indices = np.random.choice(class_indices, num_samples, replace=False)
    subset_indices.extend(selected_indices.tolist())

np.random.shuffle(subset_indices)
dataset = Subset(dataset, subset_indices)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)

classes = tuple(getattr(dataset.dataset, "classes", (
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
)))


# 2. 扩散模型核心组件（保持与原代码一致）
class UNet(nn.Module):
    """用于扩散模型的UNet骨干网络（条件生成，支持类别输入）"""

    def __init__(self, in_channels=3, num_classes=10):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, 128)

        self.down1 = self.double_conv(in_channels, 64)
        self.down2 = self.double_conv(64, 128)
        self.down3 = self.double_conv(128, 256)
        self.down4 = self.double_conv(256, 512)

        self.bottleneck = self.double_conv(512 + 128, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = self.double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = self.double_conv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = self.double_conv(128, 64)

        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t, labels):
        x1 = self.down1(x)
        x2 = self.down2(nn.MaxPool2d(2)(x1))
        x3 = self.down3(nn.MaxPool2d(2)(x2))
        x4 = self.down4(nn.MaxPool2d(2)(x3))

        c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        c = F.interpolate(c, size=x4.shape[2:], mode='bilinear', align_corners=False)

        bottleneck = self.bottleneck(torch.cat([x4, c], dim=1))

        up1 = self.up1(bottleneck)
        up1 = torch.cat([up1, x3], dim=1)
        up1 = self.conv_up1(up1)

        up2 = self.up2(up1)
        up2 = torch.cat([up2, x2], dim=1)
        up2 = self.conv_up2(up2)

        up3 = self.up3(up2)
        up3 = torch.cat([up3, x1], dim=1)
        up3 = self.conv_up3(up3)

        return self.out(up3)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Models"""

    def __init__(self, model, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super(DDPM, self).__init__()
        self.model = model.to(device)
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def forward_diffusion(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    def reverse_diffusion(self, xt, t, labels):
        return self.model(xt, t, labels)

    def loss_function(self, x0, labels):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        xt, noise = self.forward_diffusion(x0, t)
        predicted_noise = self.reverse_diffusion(xt, t, labels)
        return nn.MSELoss()(predicted_noise, noise)

    def sample(self, num_samples, labels, img_size=(3, 32, 32)):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, *img_size, device=device)
            for i in reversed(range(0, self.num_timesteps)):
                t = torch.full((num_samples,), i, device=device, dtype=torch.long)
                predicted_noise = self.reverse_diffusion(x, t, labels)

                alpha = self.alphas[t][:, None, None, None]
                alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha)) * (
                        x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(
                    beta) * noise
        self.model.train()
        x = (x + 1) / 2
        return x


# 3. 加载遗忘前后的模型并进行预测
def load_models(pre_forget_path, post_forget_path):
    """加载遗忘前和遗忘后的模型"""
    pre_model = CIFAR10CNN().to(device)
    post_model = CIFAR10CNN().to(device)

    pre_model.load_state_dict(torch.load(pre_forget_path, map_location=device))
    post_model.load_state_dict(torch.load(post_forget_path, map_location=device))

    pre_model.eval()
    post_model.eval()

    return pre_model, post_model


def predict_image(pre_model, post_model, image_tensor):
    """使用遗忘前后的模型对图像进行预测，返回完整概率分布"""
    # 图像预处理（匹配FLwithFU中的预处理方式）
    fl_transform = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 处理图像
    input_tensor = fl_transform(image_tensor.clone())
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        pre_outputs = pre_model(input_tensor)
        post_outputs = post_model(input_tensor)

        # 计算完整置信度分布（10个类别的概率向量）
        pre_probs = F.softmax(pre_outputs, dim=1).squeeze().cpu().numpy()
        post_probs = F.softmax(post_outputs, dim=1).squeeze().cpu().numpy()

        # 获取最高置信度的类别和值
        pre_max_idx = np.argmax(pre_probs)
        post_max_idx = np.argmax(post_probs)

        return {
            'pre': {
                'all_probs': pre_probs,  # 完整概率向量
                'top_class': CIFAR10_CLASSES[pre_max_idx],
                'top_confidence': pre_probs[pre_max_idx]
            },
            'post': {
                'all_probs': post_probs,  # 完整概率向量
                'top_class': CIFAR10_CLASSES[post_max_idx],
                'top_confidence': post_probs[post_max_idx]
            }
        }


def add_prediction_text(image, predictions):
    """在图像右侧添加预测结果文本（修复中文显示，展示全部类别概率分布）"""
    # 导入必要库
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch

    # 设置中文字体，解决中文显示问题
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 将PyTorch张量转换为PIL图像（处理[-1,1]到[0,255]的转换）
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # 转换通道顺序为(H,W,C)
        image = (image + 1) / 2  # 从[-1,1]归一化到[0,1]
        image = (image * 255).astype(np.uint8)  # 转换为[0,255]整数
        image = Image.fromarray(image)

    # 调整文本区域宽度以容纳全部10个类别（CIFAR10共10类）
    text_area_width = 400  # 加宽文本区域，确保全部类别显示完整
    new_width = image.width + text_area_width
    new_image = Image.new('RGB', (new_width, image.height), color='white')
    new_image.paste(image, (0, 0))

    # 计算文本布局参数（适配全部10个类别的显示）
    start_y = max(20, image.height * 0.05)  # 起始y坐标
    line_spacing = max(18, image.height * 0.04)  # 增大行间距，避免10个类别重叠
    font_size = min(8, int(image.height * 0.022))  # 微调字体大小，保证显示完整

    # 使用matplotlib添加文本
    plt.figure(figsize=(new_width / 100, image.height / 100), dpi=100)
    plt.imshow(new_image)
    plt.axis('off')  # 关闭坐标轴

    # 获取全部类别的概率（CIFAR10共10类）
    pre_probs = predictions['pre']['label']
    post_probs = predictions['post']['label']
    num_classes = len(pre_probs)  # 应为10

    # 添加遗忘前模型预测（全部类别）
    plt.text(image.width + 10, start_y, "遗忘前模型预测（全部类别）:", fontsize=font_size, fontweight='bold')
    for class_idx in range(num_classes):
        # 按类别索引顺序展示，保持一致性
        plt.text(
            image.width + 20,
            start_y + line_spacing * (class_idx + 1),  # 逐行显示
            f"{CIFAR10_CLASSES[class_idx]}: {pre_probs[class_idx]:.4f}",
            fontsize=font_size
        )

    # 添加遗忘后模型预测（全部类别），与前部分保持适当距离
    post_start_y = start_y + line_spacing * (num_classes + 2)  # 留出2行空白分隔
    plt.text(image.width + 10, post_start_y, "遗忘后模型预测（全部类别）:", fontsize=font_size, fontweight='bold')
    for class_idx in range(num_classes):
        plt.text(
            image.width + 20,
            post_start_y + line_spacing * (class_idx + 1),
            f"{CIFAR10_CLASSES[class_idx]}: {post_probs[class_idx]:.4f}",
            fontsize=font_size
        )

    # 保存到缓冲区并转换为PIL图像
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    result_image = Image.open(buf)

    # 关闭matplotlib图形，释放资源
    plt.close()

    return result_image


# 4. 训练函数（保持与原代码一致）
def train_ddpm(ddpm, dataloader, epochs=100, lr=2e-4):
    optimizer = optim.Adam(ddpm.parameters(), lr=lr)
    ddpm.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = ddpm.loss_function(images, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(ddpm.state_dict(), f"ddpm_cifar10_epoch_{epoch + 1}.pth")


# 5. 升级后的生成图像函数
def generate_images(ddpm, pre_forget_model_path, post_forget_model_path,
                    num_per_class=100, save_dir="generated_images_with_predictions"):
    os.makedirs(save_dir, exist_ok=True)
    ddpm.eval()

    # 加载遗忘前后的模型
    pre_model, post_model = load_models(pre_forget_model_path, post_forget_model_path)

    for class_idx in range(10):
        class_name = classes[class_idx]
        class_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # 生成该类别的图像
        for i in range(0, num_per_class, 32):
            batch_size = min(32, num_per_class - i)
            labels = torch.full((batch_size,), class_idx, device=device, dtype=torch.long)
            generated = ddpm.sample(batch_size, labels)

            # 处理每张图像
            for j in range(batch_size):
                img_tensor = generated[j]

                # 获取预测结果
                predictions = predict_image(pre_model, post_model, img_tensor)

                # 添加文本到图像
                image_with_text = add_prediction_text(img_tensor, predictions)

                # 保存图像
                img_path = os.path.join(class_dir, f"{i + j}.png")
                image_with_text.save(img_path)

        print(f"已生成 {class_name} 类别 {num_per_class} 张图像（含预测结果）")


# 主函数
if __name__ == "__main__":
    # 初始化模型
    unet = UNet()
    ddpm = DDPM(unet).to(device)

    # 训练模型
    print("开始预训练...")
    train_ddpm(ddpm, dataloader, epochs=100)

    # 生成图像（需要指定遗忘前后模型的路径）
    print("开始生成图像并添加预测结果...")
    # 请替换为实际的模型路径
    pre_forget_path = "model/federated_cifar10_pre_FedAF.pth"
    post_forget_path = "model/federated_cifar10_post_FedAF.pth"
    generate_images(ddpm, pre_forget_path, post_forget_path, num_per_class=100)
    print("生成完成！")