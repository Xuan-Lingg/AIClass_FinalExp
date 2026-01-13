import os
import sys
import shutil
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ==========================================
# 1. 环境设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
demo_root = os.path.dirname(current_dir)
project_root = os.path.dirname(demo_root)
sys.path.append(project_root)

# 导入模型定义
from src.model import ResNet18_Baseline, ResNet18_With_Attention
# 导入可视化工具
from src.analyze_kqv import get_attention_heatmap, overlay_heatmap

# ==========================================
# 2. 全局配置
# ==========================================
DATA_DIR = os.path.join(current_dir, 'Data')
RESULT_DIR = os.path.join(current_dir, 'result')
ORIGINAL_TRAIN_DIR = os.path.join(project_root, 'data', 'cub200', 'train')

def load_classes():
    """读取原始训练集获取类别名称"""
    if not os.path.exists(ORIGINAL_TRAIN_DIR):
        print("错误：无法找到原始训练数据，无法构建类别映射！")
        sys.exit(1)
    return sorted([d.name for d in os.scandir(ORIGINAL_TRAIN_DIR) if d.is_dir()])

def get_transform():
    return transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def clean_and_init_result_dir():
    """初始化结果目录"""
    if os.path.exists(RESULT_DIR):
        print(f"Cleaning output directory: {RESULT_DIR}")
        try:
            shutil.rmtree(RESULT_DIR)
        except Exception as e:
            print(f"Warning: Failed to delete {RESULT_DIR}: {e}")
    
    os.makedirs(os.path.join(RESULT_DIR, 'baseline'))
    os.makedirs(os.path.join(RESULT_DIR, 'attention'))
    os.makedirs(os.path.join(RESULT_DIR, 'attention', 'heatmaps'))
    print("Output directories initialized.\n")

def run_prediction(model_type, full_classes, device):
    """
    通用的预测逻辑
    Args:
        model_type: 'baseline' 或 'attention'
    """
    print(f"\n{'='*20} Running Inference: {model_type.upper()} Model {'='*20}")
    
    # 1. 模型加载逻辑
    num_classes = len(full_classes)
    weight_path = os.path.join(project_root, 'outputs', 'checkpoints', f'cub200_{model_type}.pth')
    
    if not os.path.exists(weight_path):
        print(f"Skipping {model_type}: Checkpoint not found at {weight_path}")
        return

    if model_type == 'baseline':
        model = ResNet18_Baseline(num_classes=num_classes)
        save_sub_dir = os.path.join(RESULT_DIR, 'baseline')
    else:
        model = ResNet18_With_Attention(num_classes=num_classes)
        save_sub_dir = os.path.join(RESULT_DIR, 'attention')

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transform()
    
    known_logs = []
    unknown_logs = []
    total_known = 0
    correct_known = 0

    # 2. 遍历图片
    for folder_name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(folder_path): continue
        
        # 过滤空文件夹
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not images: continue

        is_unknown = ('unknown' in folder_name.lower())
        
        header = f"\n📂 文件夹: [{folder_name}]"
        print(header)
        current_logs = unknown_logs if is_unknown else known_logs
        current_logs.append(header)
        
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            try:
                # 预处理
                raw_img = Image.open(img_path).convert('RGB')
                # 这里 input_tensor 是 [1, 3, 448, 448]
                input_tensor = transform(raw_img).unsqueeze(0).to(device)
                
                # 推理
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1)
                    top5_prob, top5_idx = torch.topk(probs, 5)
                
                top5_prob = top5_prob.cpu().numpy()[0]
                top5_idx = top5_idx.cpu().numpy()[0]
                top1_class = full_classes[top5_idx[0]]

                # ========================================================
                # 【修复核心】热力图可视化 (仅 Attention 模型)
                # ========================================================
                heatmap_msg = ""
                if model_type == 'attention':
                    # 修复点：传入 input_tensor.squeeze(0)
                    # 因为 get_attention_heatmap 内部会再次 unsqueeze
                    heatmap = get_attention_heatmap(model, input_tensor.squeeze(0), device)
                    
                    heatmap_filename = f"{folder_name}_{img_name}"
                    heatmap_path = os.path.join(save_sub_dir, 'heatmaps', heatmap_filename)
                    
                    overlay_heatmap(img_path, heatmap, heatmap_path)
                    heatmap_msg = " [Heatmap Generated]"

                # 构建日志
                print(f"   🐦 {img_name}{heatmap_msg}")
                img_log = []
                img_log.append(f"   🐦 图片: {img_name}")

                if not is_unknown:
                    total_known += 1
                    if top1_class == folder_name:
                        correct_known += 1
                        img_log.append(f"      [结果] ✅ 正确")
                    else:
                        img_log.append(f"      [结果] ❌ 错误 (真实: {folder_name})")
                
                img_log.append(f"      [预测可能性]")
                for i in range(5):
                    c_name = full_classes[top5_idx[i]]
                    p_val = top5_prob[i] * 100
                    img_log.append(f"      {'->' if i==0 else '  '} No.{i+1}: {c_name:<30} {p_val:.2f}%")
                
                img_log.append("-" * 40)
                current_logs.append("\n".join(img_log))

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    # 3. 保存报告
    report_filename = f'report_{model_type}.txt'
    report_path = os.path.join(save_sub_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"      CUB-200 预测报告: {model_type.upper()} 模型\n")
        f.write("="*60 + "\n\n")

        f.write("【第一部分：已知标签验证】\n")
        if known_logs:
            for log in known_logs: f.write(log + "\n")
            acc = 100 * correct_known / total_known if total_known > 0 else 0
            f.write(f"\n>>> 统计: 准确率 {acc:.2f}% ({correct_known}/{total_known})\n")
        else:
            f.write("(无数据)\n")
        
        f.write("\n\n【第二部分：未知数据预测】\n")
        if unknown_logs:
            for log in unknown_logs: f.write(log + "\n")
        else:
            f.write("(无数据)\n")

    print(f"✅ Report saved to: {report_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_classes = load_classes()
    print(f"Loaded {len(full_classes)} classes.")

    clean_and_init_result_dir()

    run_prediction('baseline', full_classes, device)
    run_prediction('attention', full_classes, device)

    print(f"\nAll Done! Results are in: {RESULT_DIR}")

if __name__ == '__main__':
    main()