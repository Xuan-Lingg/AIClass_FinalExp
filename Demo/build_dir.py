import os

def mirror_directory_names(source_path, target_path):
    """
    读取 source_path 下的目录名，并在 target_path 下创建同名目录。
    """
    # 1. 检查源目录是否存在
    if not os.path.exists(source_path):
        print(f"错误：源目录 '{source_path}' 不存在。")
        return

    # 2. 如果目标目录不存在，先创建目标根目录
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
            print(f"提示：目标根目录 '{target_path}' 不存在，已自动创建。")
        except OSError as e:
            print(f"错误：无法创建目标目录。原因: {e}")
            return

    # 3. 获取源目录下的所有内容
    items = os.listdir(source_path)
    count = 0

    print("--- 开始处理 ---")
    
    for item in items:
        # 拼接完整的源路径
        full_source_item_path = os.path.join(source_path, item)

        # 4. 判断是否为目录（忽略文件）
        if os.path.isdir(full_source_item_path):
            # 拼接完整的目标路径
            full_target_item_path = os.path.join(target_path, item)
            
            try:
                # 5. 创建目录 (exist_ok=True 表示如果目录已存在不仅不报错，直接跳过)
                os.makedirs(full_target_item_path, exist_ok=True)
                print(f"[成功] 已创建/确认目录: {item}")
                count += 1
            except Exception as e:
                print(f"[失败] 无法创建目录 {item}: {e}")
    
    print("--- 处理结束 ---")
    print(f"共处理了 {count} 个目录。")

if __name__ == "__main__":
    # ================= 配置区域 =================
    # 请在这里修改你的路径
    # 注意：Windows路径建议在引号前加 r，或者使用双斜杠 \\
    
    # 输入：你要读取的文件夹路径
    source_dir = r".\raw_data\cub_200\CUB_200_2011\CUB_200_2011\images"
    
    # 输出：你要生成的文件夹路径
    target_dir = r".\Demo\cub200\Data"
    # ===========================================

    mirror_directory_names(source_dir, target_dir)