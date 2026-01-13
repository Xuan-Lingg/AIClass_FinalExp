import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ==========================================
# 核心组件：多头自注意力机制 (MHSA)
# 这部分是你报告中"原理及方法分析"的重点
# ==========================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放系数 1/sqrt(d_k)

        # 定义 Q, K, V 的映射层 (使用 1x1 卷积实现)
        self.to_q = nn.Conv2d(in_channels, inner_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, inner_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, inner_dim, kernel_size=1, bias=False)

        # 输出映射层
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)  # 加个BN有助于训练稳定
        )

    def forward(self, x, return_details=False):
        """
        Args:
            x: 输入特征图 [Batch, Channel, Height, Width]
            return_details: 是否返回 KQV 矩阵用于分析
        """
        b, c, h, w = x.shape

        # 1. 生成 Q, K, V
        # 形状变换: [B, Inner_Dim, H, W] -> [B, Heads, Inner_Dim/Heads, H*W]
        q = self.to_q(x).view(b, self.heads, -1, h * w)
        k = self.to_k(x).view(b, self.heads, -1, h * w)
        v = self.to_v(x).view(b, self.heads, -1, h * w)

        # 2. 计算点积注意力 (Scaled Dot-Product Attention)
        # dots shape: [B, Heads, H*W, H*W] -> 代表每个像素与其他所有像素的相关性
        dots = torch.matmul(q.transpose(-1, -2), k) * self.scale

        # 3. Softmax 归一化，得到 Attention Map
        attn = dots.softmax(dim=-1)

        # 4. 加权求和: Attention * V
        out = torch.matmul(attn, v.transpose(-1, -2))

        # 5. 还原形状: [B, Heads, Dim, H*W] -> [B, C, H, W]
        out = out.transpose(-1, -2).contiguous().view(b, -1, h, w)

        # 6. 输出映射 + 残差连接 (Residual Connection)
        # 注意：残差连接在外部做，这里只返回变换后的特征
        final_out = self.to_out(out)

        if return_details:
            # 返回所有用于分析的中间变量
            # attn: 注意力图
            # q, k, v: 原始矩阵
            return final_out, attn, q, k, v

        return final_out


# ==========================================
# 模型一：基准 ResNet18 (Baseline)
# ==========================================
class ResNet18_Baseline(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # 加载预训练模型
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # 修改最后的全连接层，适配我们的分类数 (2或200)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ==========================================
# 模型二：ResNet18 + 自注意力 (Improved)
# ==========================================
class ResNet18_With_Attention(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)

        # 拆解 ResNet 结构
        # 0. 初始层 (conv1, bn1, relu, maxpool)
        self.initial = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )

        # 1-4. 中间残差层 (Layer 1-4)
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4  # 输出 512 通道

        # ==================================================
        # 【核心修改】在这里插入自注意力模块
        # ResNet18 layer4 的输出通道是 512
        # ==================================================
        self.attention = MultiHeadSelfAttention(in_channels=512, heads=4, dim_head=64)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的权重，初始为0，让模型逐渐学会在CNN基础上加Attention

        # 5. 结尾分类层
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_attn=False):
        # 前向传播 - 标准 ResNet 流程
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # 此时 shape 约 [B, 512, 7, 7] (如果是224输入)

        # ==================================================
        # 【插入点】自注意力机制
        # ==================================================
        if return_attn:
            # 如果是分析模式，我们需要接收 KQV
            attn_out, attn_map, q, k, v = self.attention(x, return_details=True)
            # 残差连接: Original + Gamma * Attention
            x = x + self.gamma * attn_out
        else:
            # 正常训练模式
            attn_out = self.attention(x)
            x = x + self.gamma * attn_out

        # 结尾流程
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        if return_attn:
            return logits, attn_map, q, k, v

        return logits


# ==========================================
# 测试代码 (确保维度匹配)
# ==========================================
if __name__ == "__main__":
    # 模拟输入 [Batch=2, Channels=3, H=224, W=224]
    dummy_img = torch.randn(2, 3, 224, 224)

    print("Testing Baseline Model...")
    model_base = ResNet18_Baseline(num_classes=200)
    out = model_base(dummy_img)
    print(f"Baseline Output: {out.shape}")  # 应该是 [2, 200]

    print("\nTesting Attention Model...")
    model_attn = ResNet18_With_Attention(num_classes=200)

    # 1. 正常前向传播
    out = model_attn(dummy_img)
    print(f"Normal Output: {out.shape}")

    # 2. 分析模式前向传播
    logits, attn_map, q, k, v = model_attn(dummy_img, return_attn=True)
    print("Analysis Mode Output:")
    print(f" - Logits: {logits.shape}")
    print(f" - Attention Map: {attn_map.shape}")  # [2, 4, 49, 49] (49 = 7*7)
    print(f" - Query (Q): {q.shape}")
    print("Model test passed!")