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
#print(torch.version.cuda)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"使用设备: {device}")

# 1. 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为[0,1]张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到[-1,1]
])

# 加载CIFAR-10数据集
dataset = datasets.CIFAR10(
    root='data',
    train=True,  # 使用训练集
    download=True,
    transform=transform
)

# 在保持与CIFAR-10同分布的情况下抽取10%的子集用于预训练
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

# 类别名称
classes = tuple(getattr(dataset.dataset, "classes", (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)))


# 2. 扩散模型核心组件
class UNet(nn.Module):
    """用于扩散模型的UNet骨干网络（条件生成，支持类别输入）"""

    def __init__(self, in_channels=3, num_classes=10):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        # 类别嵌入层
        self.label_emb = nn.Embedding(num_classes, 128)

        # 下采样部分
        self.down1 = self.double_conv(in_channels, 64)
        self.down2 = self.double_conv(64, 128)
        self.down3 = self.double_conv(128, 256)
        self.down4 = self.double_conv(256, 512)

        # 瓶颈层
        self.bottleneck = self.double_conv(512 + 128, 512)  # 加入类别嵌入

        # 上采样部分
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = self.double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = self.double_conv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = self.double_conv(128, 64)

        # 输出层（预测噪声）
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
        # 先进行下采样得到各层特征，确保x4被定义
        x1 = self.down1(x)
        x2 = self.down2(nn.MaxPool2d(2)(x1))
        x3 = self.down3(nn.MaxPool2d(2)(x2))
        x4 = self.down4(nn.MaxPool2d(2)(x3))  # 现在x4已经被定义

        # 处理类别嵌入并调整尺寸（使用已定义的x4）
        c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)  # (B, 128, 1, 1)
        c = F.interpolate(c, size=x4.shape[2:], mode='bilinear', align_corners=False)  # 适配x4的尺寸

        # 瓶颈层拼接
        bottleneck = self.bottleneck(torch.cat([x4, c], dim=1))

        # 上采样
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

        # 噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def forward_diffusion(self, x0, t, noise=None):
        """前向扩散：x0 -> xt"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    def reverse_diffusion(self, xt, t, labels):
        """反向扩散：预测噪声"""
        return self.model(xt, t, labels)

    def loss_function(self, x0, labels):
        """计算损失（预测噪声与真实噪声的MSE）"""
        batch_size = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        xt, noise = self.forward_diffusion(x0, t)
        predicted_noise = self.reverse_diffusion(xt, t, labels)
        return nn.MSELoss()(predicted_noise, noise)

    def sample(self, num_samples, labels, img_size=(3, 32, 32)):
        """生成图像：从噪声逐步去噪"""
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, *img_size, device=device)  # 初始噪声
            for i in reversed(range(0, self.num_timesteps)):
                t = torch.full((num_samples,), i, device=device, dtype=torch.long)
                predicted_noise = self.reverse_diffusion(x, t, labels)

                # 计算去噪参数
                alpha = self.alphas[t][:, None, None, None]
                alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # 反向步骤更新
                x = (1 / torch.sqrt(alpha)) * (
                            x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(
                    beta) * noise
        self.model.train()
        x = (x + 1) / 2  # 从[-1,1]转回[0,1]
        return x


# 3. 训练函数
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

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(ddpm.state_dict(), f"ddpm_cifar10_epoch_{epoch + 1}.pth")


# 4. 生成图像函数
def generate_images(ddpm, num_per_class=100, save_dir="generated_images"):
    os.makedirs(save_dir, exist_ok=True)
    ddpm.eval()

    for class_idx in range(10):
        class_name = classes[class_idx]
        class_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # 生成该类别的num_per_class张图像
        for i in range(0, num_per_class, 32):  # 分批生成（避免显存不足）
            batch_size = min(32, num_per_class - i)
            labels = torch.full((batch_size,), class_idx, device=device, dtype=torch.long)
            generated = ddpm.sample(batch_size, labels)

            # 保存图像
            for j in range(batch_size):
                img_path = os.path.join(class_dir, f"{i + j}.png")
                save_image(generated[j], img_path)

        print(f"已生成 {class_name} 类别 {num_per_class} 张图像")


# 主函数
if __name__ == "__main__":
    # 初始化模型
    unet = UNet()
    ddpm = DDPM(unet).to(device)

    # 训练模型（使用测试集）
    print("开始预训练...")
    train_ddpm(ddpm, dataloader, epochs=100)  # 可根据需要调整epochs

    # 生成图像（每类100张）
    print("开始生成图像...")
    generate_images(ddpm, num_per_class=100)
    print("生成完成！")