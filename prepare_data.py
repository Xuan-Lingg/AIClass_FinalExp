import os
import shutil
import random

# ================= 配置区域 =================
# 项目根目录 (获取当前脚本所在路径)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Cat_Dog 配置
# 指向包含 'Cat' 和 'Dog' 文件夹的父级目录
RAW_CAT_DOG_DIR = os.path.join(BASE_DIR, 'raw_data', 'cat_dog', 'PetImages')
TARGET_CAT_DOG_DIR = os.path.join(BASE_DIR, 'data', 'cat_dog')
SPLIT_RATIO = 0.9  # 90% 训练, 10% 验证

# 2. CUB_200 配置
# 指向包含 'images', 'images.txt' 等文件的目录
# 根据你提供的结构，它在两层 CUB_200_2011 下
RAW_CUB_DIR = os.path.join(BASE_DIR, 'raw_data', 'cub_200', 'CUB_200_2011', 'CUB_200_2011')
TARGET_CUB_DIR = os.path.join(BASE_DIR, 'data', 'cub200')


# ================= 工具函数 =================

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_files(file_list, source_dir, target_dir):
    """
    通用文件复制函数
    file_list: 文件名列表
    source_dir: 源文件所在目录
    target_dir: 目标目录
    """
    make_dir(target_dir)
    count = 0
    for filename in file_list:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(target_dir, filename)

        # 忽略非图片文件 (如 Thumbs.db)
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        try:
            shutil.copy(src, dst)
            count += 1
        except Exception as e:
            print(f"Error copying {src}: {e}")
    return count


# ================= 处理逻辑：Cat vs Dog =================

def process_cat_dog():
    print(f"\n[1/2] 正在处理 Cat_Dog 数据集...")

    if not os.path.exists(RAW_CAT_DOG_DIR):
        print(f"错误: 找不到源路径 {RAW_CAT_DOG_DIR}")
        return

    classes = ['Cat', 'Dog']

    for class_name in classes:
        source_class_dir = os.path.join(RAW_CAT_DOG_DIR, class_name)
        if not os.path.exists(source_class_dir):
            print(f"警告: 找不到类别文件夹 {source_class_dir}")
            continue

        # 获取所有图片文件
        images = os.listdir(source_class_dir)
        # 过滤掉非图片文件
        images = [x for x in images if x.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 随机打乱
        random.seed(42)  # 保证每次运行结果一致
        random.shuffle(images)

        # 计算切分点
        split_point = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_point]
        val_imgs = images[split_point:]

        # 目标路径
        train_dir = os.path.join(TARGET_CAT_DOG_DIR, 'train', class_name)
        val_dir = os.path.join(TARGET_CAT_DOG_DIR, 'val', class_name)

        print(f"正在处理 {class_name}: 总数 {len(images)} -> 训练集 {len(train_imgs)} | 验证集 {len(val_imgs)}")

        # 执行复制
        copy_files(train_imgs, source_class_dir, train_dir)
        copy_files(val_imgs, source_class_dir, val_dir)

    print("Cat_Dog 数据集处理完成！")


# ================= 处理逻辑：CUB-200 =================

def process_cub():
    print(f"\n[2/2] 正在处理 CUB-200 数据集...")

    images_txt_path = os.path.join(RAW_CUB_DIR, 'images.txt')
    split_txt_path = os.path.join(RAW_CUB_DIR, 'train_test_split.txt')
    raw_images_dir = os.path.join(RAW_CUB_DIR, 'images')

    # 检查必要文件
    if not os.path.exists(images_txt_path) or not os.path.exists(split_txt_path):
        print(f"错误: 在 {RAW_CUB_DIR} 下找不到 images.txt 或 train_test_split.txt")
        print("请检查 raw_data 目录结构是否正确。")
        return

    # 1. 读取 Image ID -> File Path 映射
    id2path = {}
    with open(images_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id2path[parts[0]] = parts[1]  # ID: '1', Path: '001.Black.../xxx.jpg'

    # 2. 读取 Image ID -> Train/Test 标记
    id2train = {}
    with open(split_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id2train[parts[0]] = int(parts[1])  # 1=Train, 0=Test

    count_train = 0
    count_val = 0

    print("正在根据官方索引整理 CUB 文件...")

    # 3. 遍历并复制
    for img_id, rel_path in id2path.items():
        # 确定是训练还是验证 (官方叫test，我们这里统一叫val方便代码管理)
        is_train = id2train.get(img_id, 0)
        mode = 'train' if is_train == 1 else 'val'

        src_file = os.path.join(raw_images_dir, rel_path)
        dst_file = os.path.join(TARGET_CUB_DIR, mode, rel_path)

        if not os.path.exists(src_file):
            continue

        make_dir(os.path.dirname(dst_file))
        shutil.copy(src_file, dst_file)

        if mode == 'train':
            count_train += 1
        else:
            count_val += 1

        if (count_train + count_val) % 1000 == 0:
            print(f"已处理 {count_train + count_val} 张图片...")

    print(f"CUB-200 处理完成: 训练集 {count_train} 张, 验证集 {count_val} 张")


if __name__ == '__main__':
    # 确保 data 目录存在
    make_dir(os.path.join(BASE_DIR, 'data'))

    # 执行处理
    process_cat_dog()
    process_cub()
    print("\n所有数据准备完毕！请检查 ./data 目录。")