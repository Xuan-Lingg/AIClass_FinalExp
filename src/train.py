# 文件路径: src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import copy
import sys
import random
from PIL import Image            # 【新增】
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.dataset import load_data
from src.model import ResNet18_Baseline, ResNet18_With_Attention
from src.utils import (
    plot_training_curves, 
    get_all_predictions, 
    plot_confusion_matrix_custom, 
    save_classification_report_txt
)
# 【新增】导入刚才写的新函数
from src.analyze_kqv import visualize_epoch_progress

# 路径配置
CHECKPOINT_DIR = './outputs/checkpoints'
LOG_DIR = './outputs/logs'
VIS_DIR = './outputs/visualizations'

def train_loop(model, dataloaders, dataset_sizes, criterion, optimizer, device, 
               num_epochs=25, save_name="best_model.pth", 
               # 【新增参数】用于可视化的固定样本
               vis_sample=None, vis_save_dir=None):
    """
    训练循环
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    time_str = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_DIR, f"{time_str}_{save_name.split('.')[0]}")
    writer = SummaryWriter(log_path)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(dataloaders[phase], desc=f"{phase:>5}", leave=True, file=sys.stdout)

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            print(f"{phase} Result -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if not os.path.exists(CHECKPOINT_DIR):
                    os.makedirs(CHECKPOINT_DIR)
                save_path = os.path.join(CHECKPOINT_DIR, save_name)
                torch.save(model.state_dict(), save_path)
                print(f"  ★ Best model saved to {save_path}")

        # ========================================================
        # 【新增逻辑】每个 Epoch 结束后，绘制 KQV 热力图
        # ========================================================
        # 只有当模型是 Attention 版本，且提供了可视化样本时才画
        if vis_sample is not None and isinstance(model, ResNet18_With_Attention):
            img_tensor, img_path = vis_sample
            visualize_epoch_progress(
                model, 
                img_tensor.to(device), # 确保 Tensor 在 GPU 上
                img_path, 
                epoch, 
                vis_save_dir, 
                device
            )

    time_elapsed = time.time() - since
    writer.close()
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

def start_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = os.path.join(os.getcwd(), 'data')
    input_size = 448 if args.dataset == 'cub200' else 224
    
    dataloaders, sizes, class_names, num_classes = load_data(
        data_dir, args.dataset, args.batch_size, input_size=input_size
    )

    print(f"Initializing {args.model} model...")
    if args.model == 'baseline':
        model = ResNet18_Baseline(num_classes=num_classes)
        save_name = f"{args.dataset}_baseline.pth"
    else:
        model = ResNet18_With_Attention(num_classes=num_classes)
        save_name = f"{args.dataset}_attention.pth"
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # ========================================================
    # 【新增准备工作】选取一张固定图片用于可视化
    # ========================================================
    vis_sample = None
    vis_save_dir = None
    
    if args.model == 'attention': # 只有 Attention 模型才需要准备这个
        print("Preparing visualization sample for KQV analysis...")
        # 1. 从验证集中随机选一张图 (或者你可以指定固定路径)
        val_dataset = dataloaders['val'].dataset
        # 这里为了演示，我们固定取验证集的第 0 张图，保证每次运行都一样，方便对比
        # 如果你想随机，可以用 random.choice(val_dataset.samples)
        sample_path, _ = val_dataset.samples[0] 
        
        print(f"Selected sample for monitoring: {sample_path}")
        
        # 2. 预处理这张图
        # 我们需要用验证集的 transform (val_dataset.transform)
        raw_img = Image.open(sample_path).convert('RGB')
        img_tensor = val_dataset.transform(raw_img) # [3, H, W]
        # 注意：这里不需要加 unsqueeze(0)，因为后面函数里可能会加，或者 train_loop 里处理
        # 修正：visualize_epoch_progress 里 get_attention_heatmap 需要 batch 维度
        # 所以我们这里不动，等传入函数时处理
        
        vis_sample = (img_tensor, sample_path)
        
        # 3. 创建带时间戳的保存目录，防止覆盖
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        # 存到 outputs/visualizations/kqv_analysis/training_xxxx
        vis_save_dir = os.path.join(VIS_DIR, 'kqv_analysis', f'training_{timestamp}')
        if not os.path.exists(vis_save_dir):
            os.makedirs(vis_save_dir)
            
        print(f"Epoch heatmaps will be saved to: {vis_save_dir}")

    # ====================================================
    # 阶段一：训练
    # ====================================================
    print("\n>>> Phase 1: Start Training")
    best_model, history = train_loop(
        model, dataloaders, sizes, criterion, optimizer, device,
        num_epochs=args.epochs, save_name=save_name,
        # 传入新增参数
        vis_sample=vis_sample, vis_save_dir=vis_save_dir
    )
    
    # ... (Phase 2 和 Phase 3 代码保持不变) ...
    print("\n>>> Phase 2: Plotting Training Curves")
    exp_vis_dir = os.path.join(VIS_DIR, f"{args.dataset}_{args.model}")
    plot_training_curves(history, exp_vis_dir)

    print("\n>>> Phase 3: Final Evaluation on Validation Set")
    y_true, y_pred = get_all_predictions(best_model, dataloaders['val'], device)
    plot_confusion_matrix_custom(y_true, y_pred, class_names, exp_vis_dir)
    save_classification_report_txt(y_true, y_pred, class_names, exp_vis_dir)

    print(f"\nAll results saved to: {exp_vis_dir}")