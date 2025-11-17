import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import io

# 导入FLwithFU中的模型和数据处理函数
from FLwithFU import CIFAR10CNN, CIFAR10_CLASSES


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC  # Pillow >= 9.1.0
except AttributeError:  # pragma: no cover - 兼容旧版本Pillow
    RESAMPLE_BICUBIC = Image.BICUBIC


def configure_font_for_display():
    """配置可用的中文字体，避免matplotlib重复的缺失字体警告。

    返回值:
        str | None: 成功配置时返回字体名称，失败返回 ``None``。
    """

    # 允许通过环境变量或项目中的字体文件进行配置
    preferred_font_paths = []
    env_font_path = os.getenv("CHINESE_FONT_PATH")
    if env_font_path and os.path.exists(env_font_path):
        preferred_font_paths.append(env_font_path)

    local_font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    for candidate in (
        "NotoSansSC-Regular.otf",
        "NotoSansCJKsc-Regular.otf",
        "SourceHanSansCN-Regular.otf",
        "SourceHanSansSC-Regular.otf",
        "SimHei.ttf",
        "msyh.ttc",
    ):
        candidate_path = os.path.join(local_font_dir, candidate)
        if os.path.exists(candidate_path):
            preferred_font_paths.append(candidate_path)

    preferred_fonts = [
        "SimHei",
        "WenQuanYi Micro Hei",
        "Heiti TC",
        "Microsoft YaHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]

    # 优先尝试显式提供的字体路径
    for font_path in preferred_font_paths:
        try:
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
        except OSError:
            continue

        plt.rcParams["font.family"] = [font_name]
        plt.rcParams["font.sans-serif"] = [font_name]
        plt.rcParams["axes.unicode_minus"] = False
        return font_name

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = [font_name]
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    # 没有合适的中文字体时，回退到默认字体（英文提示），避免重复警告
    default_family = plt.rcParams.get("font.family", ["DejaVu Sans"])  # type: ignore[arg-type]
    plt.rcParams["font.family"] = default_family
    plt.rcParams["axes.unicode_minus"] = False
    return None

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

subset_ratio = 0.05
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

def cosine_beta_schedule(timesteps, s=0.008):
    """Improved DDPM cosine beta schedule."""

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, min=1e-4, max=0.999).float()


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.activation = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.activation(self.norm1(x)))

        time_out = self.time_proj(self.activation(time_emb))
        h = h + time_out[:, :, None, None]

        h = self.conv2(self.activation(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(c), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


# 2. 改进版扩散模型核心组件
class UNet(nn.Module):
    """Improved DDPM UNet backbone for CIFAR-10."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_channels: int = 128,
                 channel_mults=(1, 2, 2, 4), num_res_blocks: int = 2):
        super().__init__()
        self.num_classes = num_classes
        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.label_emb = nn.Embedding(num_classes, time_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        in_ch = base_channels
        attention_resolutions = {1, 2}
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, out_ch, time_dim))
                in_ch = out_ch
            attn = AttentionBlock(out_ch) if i in attention_resolutions else None
            downsample = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1) if i < len(channel_mults) - 1 else None
            self.downs.append(nn.ModuleDict({
                "blocks": nn.ModuleList(blocks),
                "attn": attn,
                "down": downsample
            }))

        self.bottleneck = nn.ModuleDict({
            "res1": ResidualBlock(in_ch, in_ch, time_dim),
            "attn": AttentionBlock(in_ch),
            "res2": ResidualBlock(in_ch, in_ch, time_dim)
        })

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            first_block = ResidualBlock(in_ch + out_ch, out_ch, time_dim)
            additional_blocks = [ResidualBlock(out_ch, out_ch, time_dim) for _ in range(num_res_blocks)]
            attn = AttentionBlock(out_ch) if i in attention_resolutions else None
            upsample = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1) if i > 0 else None
            self.ups.append(nn.ModuleDict({
                "blocks": nn.ModuleList([first_block] + additional_blocks),
                "attn": attn,
                "up": upsample
            }))
            in_ch = out_ch

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t)
        time_emb = time_emb + self.label_emb(labels)

        h = self.init_conv(x)
        skips = []
        for down in self.downs:
            for block in down["blocks"]:
                h = block(h, time_emb)
            if down["attn"] is not None:
                h = down["attn"](h)
            skips.append(h)
            if down["down"] is not None:
                h = down["down"](h)

        h = self.bottleneck["res1"](h, time_emb)
        h = self.bottleneck["attn"](h)
        h = self.bottleneck["res2"](h, time_emb)

        for up in self.ups:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in up["blocks"]:
                h = block(h, time_emb)
            if up["attn"] is not None:
                h = up["attn"](h)
            if up["up"] is not None:
                h = up["up"](h)

        return self.out(h)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Models"""

    def __init__(self, model: nn.Module, num_timesteps: int = 1000):
        super(DDPM, self).__init__()
        self.model = model.to(device)
        self.num_timesteps = num_timesteps

        betas = cosine_beta_schedule(num_timesteps)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.log_variance = nn.Parameter(torch.log(betas.clone()))

    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    def reverse_diffusion(self, xt: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.model(xt, t, labels)

    def loss_function(self, x0: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        xt, noise = self.forward_diffusion(x0, t)
        predicted_noise = self.reverse_diffusion(xt, t, labels)
        mse_loss = nn.MSELoss()(predicted_noise, noise)
        variance_target = torch.log(self.betas[t])
        variance_loss = 1e-3 * F.mse_loss(self.log_variance[t], variance_target)
        return mse_loss + variance_loss

    def sample(self, num_samples: int, labels: torch.Tensor, img_size=(3, 32, 32)) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, *img_size, device=device)
            for i in reversed(range(0, self.num_timesteps)):
                t = torch.full((num_samples,), i, device=device, dtype=torch.long)
                predicted_noise = self.reverse_diffusion(x, t, labels)

                alpha = self.alphas[t][:, None, None, None]
                alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
                variance = torch.exp(self.log_variance[t])[:, None, None, None]

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha)) * (
                        x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(
                    variance) * noise
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

        # 构造（类别, 概率）列表，排除概率为0的类别
        def build_prob_list(prob_array):
            prob_list = [
                (CIFAR10_CLASSES[idx], float(prob))
                for idx, prob in enumerate(prob_array)
                if float(prob) > 0.0
            ]
            if not prob_list:
                # 极端情况下若全部概率为0，回退到完整列表
                prob_list = [
                    (CIFAR10_CLASSES[idx], float(prob_array[idx]))
                    for idx in range(len(prob_array))
                ]
            return prob_list

        pre_prob_list = build_prob_list(pre_probs)
        post_prob_list = build_prob_list(post_probs)

        # 获取最高置信度的类别和值
        pre_max_idx = int(np.argmax(pre_probs))
        post_max_idx = int(np.argmax(post_probs))

        return {
            'pre': {
                'probabilities': pre_prob_list,
                'top_class': CIFAR10_CLASSES[pre_max_idx],
                'top_confidence': float(pre_probs[pre_max_idx])
            },
            'post': {
                'probabilities': post_prob_list,
                'top_class': CIFAR10_CLASSES[post_max_idx],
                'top_confidence': float(post_probs[post_max_idx])
            }
        }


def add_prediction_text(image, predictions, scale_factor=1.5):
    """在图像右侧添加预测结果文本，并根据需要等比例放大输出。

    参数:
        image (PIL.Image.Image | torch.Tensor): 原始图像或张量。
        predictions (dict): 遗忘前后模型的预测结果。
        scale_factor (float): 图像与文本的整体缩放比例，默认 ``1.5``。
    """

    font_name = configure_font_for_display()
    use_chinese_headings = font_name is not None

    # 将PyTorch张量转换为PIL图像（处理[-1,1]到[0,255]的转换）
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # 转换通道顺序为(H,W,C)
        image = (image + 1) / 2  # 从[-1,1]归一化到[0,1]
        image = (image * 255).astype(np.uint8)  # 转换为[0,255]整数
        image = Image.fromarray(image)

    scale_factor = max(scale_factor, 1.0)
    scaled_width = int(round(image.width * scale_factor))
    scaled_height = int(round(image.height * scale_factor))
    if scale_factor != 1.0:
        image = image.resize((scaled_width, scaled_height), resample=RESAMPLE_BICUBIC)
    else:
        scaled_width, scaled_height = image.width, image.height

    # 调整文本区域宽度以容纳全部10个类别（CIFAR10共10类）
    text_area_width = int(round(400 * scale_factor))  # 加宽文本区域，确保全部类别显示完整
    new_width = scaled_width + text_area_width
    new_image = Image.new('RGB', (new_width, scaled_height), color='white')
    new_image.paste(image, (0, 0))

    # 计算文本布局参数（适配全部10个类别的显示）
    start_y = max(int(round(20 * scale_factor)), int(scaled_height * 0.05))  # 起始y坐标
    line_spacing = max(int(round(20 * scale_factor)), int(scaled_height * 0.045))  # 增大行间距，避免10个类别重叠
    font_size = max(int(round(14 * scale_factor)), int(scaled_height * 0.026))  # 字体大小，保证可读性

    # 使用matplotlib添加文本
    fig, ax = plt.subplots(figsize=(new_width / 100, scaled_height / 100), dpi=100)
    ax.imshow(new_image)
    ax.axis('off')  # 关闭坐标轴

    # 获取全部类别的概率（CIFAR10共10类）
    pre_probs = predictions['pre']['probabilities']
    post_probs = predictions['post']['probabilities']
    num_classes = len(pre_probs)  # CIFAR10中应显示10个类别

    # 添加遗忘前模型预测（全部类别）
    pre_heading = "遗忘前模型预测（全部类别）:" if use_chinese_headings else "Pre-unlearning predictions (all classes):"
    ax.text(
        scaled_width + 10,
        start_y,
        pre_heading,
        fontsize=font_size,
        fontweight='bold',
        va='top'
    )
    for class_idx, (class_name, prob) in enumerate(pre_probs):
        ax.text(
            scaled_width + 20,
            start_y + line_spacing * (class_idx + 1),  # 逐行显示
            f"{class_name}: {prob:.4f}",
            fontsize=font_size,
            va='top'
        )

    # 添加遗忘后模型预测（全部类别），与前部分保持适当距离
    post_start_y = start_y + line_spacing * (num_classes + 2)  # 留出2行空白分隔
    post_heading = "遗忘后模型预测（全部类别）:" if use_chinese_headings else "Post-unlearning predictions (all classes):"
    ax.text(
        scaled_width + 10,
        post_start_y,
        post_heading,
        fontsize=font_size,
        fontweight='bold',
        va='top'
    )
    for class_idx, (class_name, prob) in enumerate(post_probs):
        ax.text(
            scaled_width + 20,
            post_start_y + line_spacing * (class_idx + 1),
            f"{class_name}: {prob:.4f}",
            fontsize=font_size,
            va='top'
        )

    # 保存到缓冲区并转换为PIL图像
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    result_image = Image.open(buf)

    # 关闭matplotlib图形，释放资源
    plt.close(fig)

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
                    num_per_class=100, save_dir="generated_images_with_predictions",
                    scale_factor=1.5):
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
                image_with_text = add_prediction_text(img_tensor, predictions, scale_factor=scale_factor)

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

