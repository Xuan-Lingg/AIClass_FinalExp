# 文件路径: src/analyze_kqv.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

# 引入你的模型定义
from src.model import ResNet18_With_Attention

def get_attention_heatmap(model, img_tensor, device):
    """
    输入一张图片张量，获取模型生成的 Attention Map 热力图
    """
    model.eval()
    with torch.no_grad():
        # 1. 前向传播，开启 return_attn=True
        # model 返回: logits, attn_map, q, k, v
        # attn_map shape: [Batch, Heads, N, N], 其中 N = H*W
        _, attn_map, q, k, v = model(img_tensor.unsqueeze(0).to(device), return_attn=True)
        
        # 2. 处理 Attention Map
        # 现在的 attn_map 是 [1, 4, 49, 49] (假设特征图 7x7)
        # 我们对所有 Head 取平均 -> [1, 49, 49]
        attn_avg = torch.mean(attn_map, dim=1).squeeze(0) # [N, N]
        
        # 3. 计算“全局影响力” (Global Influence)
        # 逻辑：对于每一个 Key (列)，求所有 Query (行) 对它的关注度之和
        # 含义：这个像素被多少人“盯着看”？被盯得越多，说明它越重要。
        importance_map = torch.sum(attn_avg, dim=0) # [N] -> [49]
        
        # 4. 恢复成 2D 形状
        # N = H_feat * W_feat. 也就是说 49 = 7 * 7
        dim = int(np.sqrt(importance_map.shape[0]))
        heatmap = importance_map.view(dim, dim).cpu().numpy()
        
        # 5. 归一化到 0-1 之间，方便画图
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap

def overlay_heatmap(original_img_path, heatmap, save_path):
    """
    将低分辨率的热力图叠加到高分辨率的原图上
    """
    # 读取原图
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (224, 224)) # 统一大小
    
    # 将 heatmap (比如 7x7) 放大到原图大小 (224x224)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # 转换为伪彩色 (热力图颜色: 蓝->红)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 叠加: 原图 * 0.6 + 热力图 * 0.4
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    
    # 保存
    cv2.imwrite(save_path, superimposed_img)
    print(f"Heatmap saved to: {save_path}")

def run_analysis(dataset_name='cat_dog', model_path=None):
    # 1. 准备环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 确定数据集路径 (用于随机抽图)
    data_root = os.path.join(os.getcwd(), 'data', dataset_name, 'val')
    # 随机找一个类别文件夹
    class_name = random.choice(os.listdir(data_root))
    class_dir = os.path.join(data_root, class_name)
    # 随机找一张图
    img_name = random.choice(os.listdir(class_dir))
    img_path = os.path.join(class_dir, img_name)
    
    print(f"Analyzing image: {img_path}")
    
    # 3. 准备模型
    # 注意：这里 num_classes 需要和你训练时一致
    # 我们可以通过文件夹数量自动判断
    num_classes = len(os.listdir(data_root))
    print(f"Loading model for {num_classes} classes...")
    
    model = ResNet18_With_Attention(num_classes=num_classes, pretrained=False)
    
    if model_path is None:
        # 默认找 attention 模型的权重
        model_path = f'./outputs/checkpoints/{dataset_name}_attention.pth'
        
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Checkpoint not found at {model_path}")
        print("Please train the 'attention' model first!")
        return

    model = model.to(device)

    # 4. 图片预处理 (必须和训练时的一模一样)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 如果是鸟类 CUB200，训练时用了 448，这里也要改 448
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    raw_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(raw_img)

    # 5. 获取热力图
    heatmap = get_attention_heatmap(model, input_tensor, device)

    # 6. 保存结果
    save_dir = './outputs/visualizations/kqv_analysis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_name = f"{dataset_name}_{class_name}_{img_name.split('.')[0]}_heatmap.jpg"
    overlay_heatmap(img_path, heatmap, os.path.join(save_dir, save_name))

def visualize_epoch_progress(model, img_tensor, original_img_path, epoch, save_dir, device):
    """
    【新增函数】专门用于训练过程中，每一轮 Epoch 结束后调用。
    Args:
        model: 当前正在训练的模型对象 (在内存中)
        img_tensor: 预处理好的图片张量 [1, 3, H, W]
        original_img_path: 原图路径 (用于叠加)
        epoch: 当前轮数
        save_dir: 保存目录
        device: CPU/GPU
    """
    model.eval() # 切换到评估模式 (非常重要，否则BN层会变)
    
    # 1. 确保目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    try:
        # 2. 获取热力图 (复用之前的逻辑)
        # 注意: img_tensor 已经在 train.py 里 unsqueeze 过了，这里不需要再 unsqueeze
        heatmap = get_attention_heatmap(model, img_tensor, device)
        
        # 3. 命名文件: epoch_01.jpg, epoch_02.jpg ... 方便排序
        save_name = f"epoch_{epoch+1:03d}.jpg"
        save_path = os.path.join(save_dir, save_name)
        
        # 4. 叠加并保存
        overlay_heatmap(original_img_path, heatmap, save_path)
        # print(f"Saved epoch heatmap to {save_path}") # 嫌刷屏可以注释掉
        
    except Exception as e:
        print(f"Warning: Failed to visualize epoch {epoch}: {e}")
    
    model.train() # ！！！画完图一定要切回训练模式，否则影响后续训练！！！

if __name__ == '__main__':
    # 你可以在这里修改参数
    # 如果要分析鸟类，把 'cat_dog' 改为 'cub200'
    run_analysis(dataset_name='cat_dog')