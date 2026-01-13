# 文件路径: ./main.py
import argparse
import sys
import os
from src.train import start_train
from src.analyze_kqv import run_analysis  # 【新增导入】

# 确保 src 目录在 python 的搜索路径中 (通常运行当前目录脚本时默认就在，但为了保险)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import start_train


def main():
    # 1. 定义命令行参数
    parser = argparse.ArgumentParser(description='AI Final Project: ResNet with Self-Attention')
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'analyze'],
                        help='Choose mode: train (start training) or analyze (visualize KQV)')
    # 数据集选择
    parser.add_argument('--dataset', type=str, default='cat_dog', choices=['cat_dog', 'cub200'],
                        help='Choose dataset: cat_dog (for testing) or cub200 (for final exp)')

    # 模型选择
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'attention'],
                        help='Choose model: baseline (ResNet18) or attention (ResNet18 + MHSA)')

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    if args.mode == 'train':
        # 运行训练逻辑
        start_train(args)
    
    elif args.mode == 'analyze':
        # 运行分析逻辑
        # 注意：分析模式只针对 attention 模型，因为它才有 KQV
        if args.model == 'baseline':
            print("Error: Baseline ResNet has no Self-Attention mechanism to analyze.")
            print("Please use --model attention")
            return
            
        # 这里的 dataset 决定了去哪里抽图，以及加载哪个权重
        run_analysis(dataset_name=args.dataset)

if __name__ == '__main__':
    main()