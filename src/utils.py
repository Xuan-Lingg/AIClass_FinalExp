import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_dir):
    """
    绘制训练过程的 Loss 和 Accuracy 曲线，并保存为图片。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    # 1. 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r--', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Graph saved: {save_path}")

def get_all_predictions(model, dataloader, device):
    """
    跑一遍测试集，收集所有的 真实标签(y_true) 和 预测标签(y_pred)
    """
    model.eval()
    all_preds = []
    all_labels = []

    print("Evaluating model on validation set...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

def plot_confusion_matrix_custom(y_true, y_pred, class_names, save_dir):
    """
    绘制并保存混淆矩阵：
    1. 保存为 PNG 热力图
    2. 【新增】保存为 CSV 表格文件
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 计算混淆矩阵 (返回的是 numpy array)
    cm = confusion_matrix(y_true, y_pred)
    
    # ==========================================
    # 【新增】 保存为 CSV 文件
    # ==========================================
    # 将矩阵转换为 DataFrame，方便查看具体数值
    # index是真实标签(行)，columns是预测标签(列)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    csv_path = os.path.join(save_dir, 'confusion_matrix.csv')
    df_cm.to_csv(csv_path, encoding='utf-8')
    print(f"Confusion Matrix CSV saved: {csv_path}")

    # ==========================================
    # 绘制图片
    # ==========================================
    if len(class_names) > 20:
        plt.figure(figsize=(20, 20)) # 针对 CUB200 这种多分类的大图
    else:
        plt.figure(figsize=(10, 8))  # 针对 CatDog 的小图

    # 使用 Seaborn 画热力图
    # annot=True: 在格子里显示数字
    # fmt='d': 数字格式为整数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    img_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(img_path)
    plt.close()
    print(f"Confusion Matrix Image saved: {img_path}")

def save_classification_report_txt(y_true, y_pred, class_names, save_dir):
    """
    生成包含 Precision, Recall, F1-score 的文本报告并保存
    """
    # zero_division=0 防止某些从未被预测到的类别导致报错
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    
    print("\nClassification Report:")
    print(report)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print(f"Report saved to {save_dir}/classification_report.txt")