import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

# 1. 解决 "Image file is truncated" 报错问题
# Cat_Dog 数据集里有些图片没下载完，PIL 默认会报错，这行代码允许加载截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 2. 定义 ImageNet 的标准化参数 (因为我们后面会用预训练的 ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(input_size=224):
    """
    定义数据预处理管道
    input_size:
      - Cat_Dog 通常用 224
      - CUB200 细粒度分类建议用 448 (如果显存不够就改回 224)
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),  # 先放缩得稍大一点
            transforms.RandomResizedCrop(input_size),  # 随机裁剪
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 简单的颜色抖动
            transforms.ToTensor(),  # 转为 Tensor [0.0, 1.0]
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)  # 标准化
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.CenterCrop(input_size),  # 验证集必须中心裁剪，保证确定性
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
    }
    return data_transforms


def safe_loader(path):
    """
    自定义图片加载器：如果图片损坏，返回一张全黑图，避免程序崩溃。
    注意：在 ImageFolder 中使用 loader 参数
    """
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f"Warning: Error loading image {path} - {e}")
        # 返回一张黑色的图片作为占位符 (尺寸随意，transforms会调整它)
        return Image.new('RGB', (224, 224), (0, 0, 0))


def load_data(data_root, dataset_name, batch_size=32, input_size=224, num_workers=4):
    """
    主加载函数
    Args:
        data_root: 项目根目录下的 data 文件夹路径 (例如 './data')
        dataset_name: 'cat_dog' 或 'cub200'
        batch_size: 批次大小
        input_size: 图片输入尺寸
        num_workers: 加载线程数 (Windows建议设为0或1，Linux设为4)
    Returns:
        dataloaders: 字典 {'train': loader, 'val': loader}
        dataset_sizes: 字典 {'train': int, 'val': int}
        class_names: 列表，类别名称
        num_classes: 类别数量
    """

    # 拼接具体的数据集路径
    # 例如 ./data/cat_dog
    target_dir = os.path.join(data_root, dataset_name)

    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Dataset not found at {target_dir}. Please run prepare_data.py first.")

    # 以此获取 transforms
    transforms_dict = get_transforms(input_size)

    image_datasets = {}

    for split in ['train', 'val']:
        split_dir = os.path.join(target_dir, split)

        # 使用 ImageFolder 加载
        # loader=safe_loader 增加了对坏图的鲁棒性
        image_datasets[split] = datasets.ImageFolder(
            root=split_dir,
            transform=transforms_dict[split],
            loader=safe_loader
        )

    # 封装成 DataLoader
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'train'),  # 只有训练集需要打乱
            num_workers=num_workers,
            pin_memory=True  # 加速 GPU 传输
        )
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"Loaded {dataset_name} dataset:")
    print(f"  - Classes: {num_classes} {class_names[:5]}...")  # 只打印前5个类别防止刷屏
    print(f"  - Train images: {dataset_sizes['train']}")
    print(f"  - Val images:   {dataset_sizes['val']}")

    return dataloaders, dataset_sizes, class_names, num_classes


# ================= 测试代码 =================
# 如果直接运行此文件，将执行简单的测试
if __name__ == '__main__':
    # 假设 data 文件夹在上一级目录的 data 中
    # 注意：这里的相对路径取决于你在哪里运行命令
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    try:
        loaders, sizes, classes, n_cls = load_data(root_dir, 'cat_dog', batch_size=4)

        # 获取一个 Batch 看看长什么样
        inputs, labels = next(iter(loaders['train']))
        print(f"Batch Shape: {inputs.shape}")  # 应该是 [4, 3, 224, 224]
        print(f"Labels: {labels}")
        print("Dataset.py test passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        print("Please check if './data/cat_dog' exists.")