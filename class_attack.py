import argparse
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from FLwithFU import CIFAR10_CLASSES, CIFAR10CNN
from generate_sample import DDPM, UNet, add_prediction_text, configure_font_for_display


# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _classifier_preprocess(image_tensor: torch.Tensor) -> torch.Tensor:
    """匹配联邦分类模型的归一化方式。"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=image_tensor.device).view(3, 1, 1)
    return (image_tensor - mean) / std


def compute_gradient_similarity(pre_model: torch.nn.Module, post_model: torch.nn.Module,
                                image_tensor: torch.Tensor, target_class: int) -> float:
    """计算遗忘前后模型针对同一输入的梯度余弦相似度。"""
    # 遗忘前梯度
    pre_input = _classifier_preprocess(image_tensor.clone()).unsqueeze(0).to(DEVICE)
    pre_input.requires_grad_(True)
    pre_logit = F.log_softmax(pre_model(pre_input), dim=1)[0, target_class]
    pre_logit.backward(retain_graph=True)
    grad_pre = pre_input.grad.detach().flatten()

    # 遗忘后梯度
    post_input = _classifier_preprocess(image_tensor.clone()).unsqueeze(0).to(DEVICE)
    post_input.requires_grad_(True)
    post_logit = F.log_softmax(post_model(post_input), dim=1)[0, target_class]
    post_logit.backward(retain_graph=True)
    grad_post = post_input.grad.detach().flatten()

    # 避免零向量导致的nan
    if torch.allclose(grad_pre, torch.zeros_like(grad_pre)) or torch.allclose(grad_post, torch.zeros_like(grad_post)):
        return 0.0

    return float(F.cosine_similarity(grad_pre, grad_post, dim=0).item())


def compute_class_score(pre_model: torch.nn.Module, post_model: torch.nn.Module,
                        image_tensor: torch.Tensor, target_class: int, lambda_grad: float) -> Dict[str, float]:
    """根据s_class定义对样本打分。"""
    with torch.no_grad():
        input_tensor = _classifier_preprocess(image_tensor.clone()).unsqueeze(0).to(DEVICE)
        pre_logprob = F.log_softmax(pre_model(input_tensor), dim=1)[0, target_class]
        post_logprob = F.log_softmax(post_model(input_tensor), dim=1)[0, target_class]
        slogit = float(pre_logprob.item() - post_logprob.item())

    sgrad = compute_gradient_similarity(pre_model, post_model, image_tensor, target_class)
    total_score = slogit + lambda_grad * sgrad
    return {"slogit": slogit, "sgrad": sgrad, "sclass": total_score}


def select_top_samples(samples: List[Tuple[torch.Tensor, Dict[str, float]]], top_k: int) -> List[Tuple[torch.Tensor, Dict[str, float]]]:
    """按照s_class得分选出Top-K样本。"""
    sorted_samples = sorted(samples, key=lambda item: item[1]["sclass"], reverse=True)
    return sorted_samples[:top_k]


def save_scored_samples(samples: List[Tuple[torch.Tensor, Dict[str, float]]], pre_model: torch.nn.Module,
                        post_model: torch.nn.Module, output_dir: str, class_idx: int, scale_factor: float = 1.5) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for idx, (img_tensor, score_dict) in enumerate(samples):
        with torch.no_grad():
            pre_logits = pre_model(_classifier_preprocess(img_tensor).unsqueeze(0))
            post_logits = post_model(_classifier_preprocess(img_tensor).unsqueeze(0))
            pre_probs = F.softmax(pre_logits, dim=1).squeeze().cpu().numpy()
            post_probs = F.softmax(post_logits, dim=1).squeeze().cpu().numpy()

        predictions = {
            "pre": {
                "probabilities": list(zip(CIFAR10_CLASSES, [float(p) for p in pre_probs])),
                "top_class": CIFAR10_CLASSES[int(np.argmax(pre_probs))],
                "top_confidence": float(np.max(pre_probs)),
            },
            "post": {
                "probabilities": list(zip(CIFAR10_CLASSES, [float(p) for p in post_probs])),
                "top_class": CIFAR10_CLASSES[int(np.argmax(post_probs))],
                "top_confidence": float(np.max(post_probs)),
            },
        }
        image_with_text = add_prediction_text(img_tensor, predictions, scale_factor=scale_factor)
        image_with_text.save(os.path.join(output_dir, f"class_{class_idx}_top_{idx}_score_{score_dict['sclass']:.3f}.png"))


def visualize_score_distribution(samples: List[Tuple[torch.Tensor, Dict[str, float]]], target_class: int, save_path: str) -> None:
    scores = [item[1]["sclass"] for item in samples]
    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=30, color="steelblue", alpha=0.8)
    plt.xlabel("s_class score")
    plt.ylabel("Frequency")
    plt.title(f"Class {CIFAR10_CLASSES[target_class]} score distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_class_attack(ddpm_checkpoint: str, pre_model_path: str, post_model_path: str, num_samples_per_class: int,
                     lambda_grad: float, top_k: int, output_dir: str, num_timesteps: int = 1000) -> None:
    """按照类级联邦遗忘攻击流程生成、评分并保存样本。"""
    os.makedirs(output_dir, exist_ok=True)
    configure_font_for_display()

    # 加载扩散模型与分类器
    unet = UNet()
    ddpm = DDPM(unet, num_timesteps=num_timesteps).to(DEVICE)
    ddpm.load_state_dict(torch.load(ddpm_checkpoint, map_location=DEVICE))
    ddpm.eval()

    pre_model = CIFAR10CNN().to(DEVICE)
    pre_model.load_state_dict(torch.load(pre_model_path, map_location=DEVICE))
    pre_model.eval()

    post_model = CIFAR10CNN().to(DEVICE)
    post_model.load_state_dict(torch.load(post_model_path, map_location=DEVICE))
    post_model.eval()

    for class_idx in range(len(CIFAR10_CLASSES)):
        class_name = CIFAR10_CLASSES[class_idx]
        class_dir = os.path.join(output_dir, f"class_{class_idx}_{class_name}")
        os.makedirs(class_dir, exist_ok=True)
        collected: List[Tuple[torch.Tensor, Dict[str, float]]] = []

        # 批量采样
        batch_size = 32
        remaining = num_samples_per_class
        while remaining > 0:
            current_bs = min(batch_size, remaining)
            labels = torch.full((current_bs,), class_idx, device=DEVICE, dtype=torch.long)
            generated = ddpm.sample(current_bs, labels)
            for img in generated:
                score_dict = compute_class_score(pre_model, post_model, img, class_idx, lambda_grad)
                collected.append((img.detach().cpu(), score_dict))
            remaining -= current_bs

        # 可视化与保存
        visualize_score_distribution(collected, class_idx, os.path.join(class_dir, "score_distribution.png"))
        top_samples = select_top_samples(collected, top_k=top_k)
        save_scored_samples(top_samples, pre_model, post_model, class_dir, class_idx)

        # 保存top-k网格图方便浏览
        grid = torch.stack([img for img, _ in top_samples])
        grid = torch.clamp(grid, 0, 1)
        fig, axes = plt.subplots(1, min(top_k, 8), figsize=(min(top_k, 8) * 2, 2))
        for idx, ax in enumerate(np.atleast_1d(axes)):
            if idx >= len(grid):
                ax.axis('off')
                continue
            img_np = grid[idx].permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            ax.axis('off')
            ax.set_title(f"score={top_samples[idx][1]['sclass']:.2f}")
        plt.tight_layout()
        fig.savefig(os.path.join(class_dir, "top_samples_preview.png"))
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-level federated unlearning attack evaluator")
    parser.add_argument("--ddpm-checkpoint", required=True, help="路径：改进DDPM权重")
    parser.add_argument("--pre-model", required=True, help="遗忘前全局模型路径")
    parser.add_argument("--post-model", required=True, help="遗忘后全局模型路径")
    parser.add_argument("--samples-per-class", type=int, default=64, help="每个类别生成的样本数量")
    parser.add_argument("--lambda-grad", type=float, default=0.5, help="s_class中梯度项系数")
    parser.add_argument("--top-k", type=int, default=16, help="保存得分最高的样本数量")
    parser.add_argument("--output-dir", default="class_attack_outputs", help="输出目录")
    parser.add_argument("--timesteps", type=int, default=1000, help="扩散步数")

    args = parser.parse_args()
    run_class_attack(
        ddpm_checkpoint=args.ddpm_checkpoint,
        pre_model_path=args.pre_model,
        post_model_path=args.post_model,
        num_samples_per_class=args.samples_per_class,
        lambda_grad=args.lambda_grad,
        top_k=args.top_k,
        output_dir=args.output_dir,
        num_timesteps=args.timesteps,
    )
