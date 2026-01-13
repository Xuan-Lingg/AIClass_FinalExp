1. 这份 `README.md` 经过重新梳理，**重点突出了“实验结果存放在哪里”以及“结果文件包含什么内容”**，方便您在查看实验数据或撰写报告时快速定位。

   您可以直接复制以下内容覆盖原文件。

   ---

   # 基于自注意力机制改进 ResNet18 的图像分类与可视化系统

   ## 1. 项目简介 (Introduction)
   本项目旨在探究**自注意力机制 (Self-Attention)** 对传统卷积神经网络 (CNN) 的性能优化与可解释性增强。项目完全从零构建，以 **ResNet18** 为基准模型，通过在深层网络嵌入**多头自注意力模块 (MHSA)**，解决了 CNN 感受野受限的问题。

   实验涵盖两个不同难度的任务：
   1.  **猫狗二分类 (Cat-Dog)**：用于验证模型收敛性及可视化效果。
   2.  **CUB-200 鸟类细粒度分类**：用于验证模型在复杂背景下对微细特征（如喙、纹理）的捕捉能力。

   ---

   ## 2. 📂 实验结果与输出说明 (Key Outputs)

   **这是本项目最核心的产出部分，所有运行结果均按结构化存储。**

   ### 2.1 训练阶段结果 (`outputs/`)
   运行 `main.py` 训练后，结果将自动保存在以下位置：

   | 目录路径                                   | 文件名                      | 内容说明                                                     |
   | :----------------------------------------- | :-------------------------- | :----------------------------------------------------------- |
   | **`outputs/checkpoints/`**                 | `*.pth`                     | **模型权重**。保存了验证集准确率最高的模型参数（如 `cub200_attention.pth`）。 |
   | **`outputs/visualizations/`**              | `training_curves.png`       | **训练曲线图**。包含 Loss（损失）和 Accuracy（准确率）随 Epoch 变化的折线图。 |
   |                                            | `confusion_matrix.png`      | **混淆矩阵热力图**。展示模型在各类别上的误判情况。           |
   |                                            | `classification_report.txt` | **详细评估报告**。包含每一类的 Precision, Recall, F1-Score 指标。 |
   | **`outputs/visualizations/kqv_analysis/`** | `training_时间戳/`          | **KQV 演变热力图**。记录了模型从 **Epoch 1 到 Epoch N** 对同一张测试图关注点的变化过程（从弥散到聚焦）。 |
   | **`outputs/logs/`**                        | `events.out.tfevents...`    | **TensorBoard 日志**。用于在 TensorBoard 中查看训练动态。    |

   ### 2.2 Demo 演示阶段结果 (`Demo/.../result/`)
   运行 `Demo` 目录下的预测脚本后，结果将保存在对应的 `result` 文件夹中：

   | 目录路径                   | 子文件夹     | 内容说明                                                     |
   | :------------------------- | :----------- | :----------------------------------------------------------- |
   | **`Demo/cat_dog/result/`** | `baseline/`  | **基准模型报告**。包含 `report_baseline.txt`，记录每张图的预测结果与置信度。 |
   |                            | `attention/` | **改进模型报告**。包含 `report_attention.txt` 及 **`heatmaps/`** 文件夹（每张测试图的注意力热力图）。 |
   | **`Demo/cub200/result/`**  | `baseline/`  | **基准模型报告**。`report_baseline.txt`，包含 Top-1 和 Top-5 预测概率。 |
   |                            | `attention/` | **改进模型报告**。`report_attention.txt` 及 **`heatmaps/`** 文件夹（含红蓝热力图，展示鸟喙/眼睛等高响应区）。 |

   ---

   ## 3. 项目目录结构 (Project Structure)

   ```text
   AI_Final_Project/
   ├── main.py                    # [主程序] 统一控制训练、评估与分析的入口
   ├── prepare_data.py            # [数据脚本] 原始数据清洗、划分与格式化
   ├── requirements.txt           # [依赖文件] 项目运行所需的 Python 库
   │
   ├── src/                       # [源代码核心层]
   │   ├── dataset.py             # 数据集加载 (含 RobustImageFolder 防报错机制)
   │   ├── model.py               # 定义 ResNet18_Baseline 及 ResNet18_With_Attention
   │   ├── train.py               # 训练循环、验证逻辑、Checkpoint 保存及可视化调用
   │   ├── analyze_kqv.py         # KQV 热力图生成与叠加算法核心
   │   └── utils.py               # 绘图工具 (Loss曲线、混淆矩阵、分类报告)
   │
   ├── Demo/                      # [演示模块] 独立的应用推理模块
   │   ├── cat_dog/
   │   │   ├── predict_cat_dog.py # 猫狗预测脚本
   │   │   ├── Data/              # 测试数据 (需手动放入 Cat/Dog/unknown)
   │   │   └── result/            # [输出] 存放预测报告和热力图
   │   └── cub200/
   │       ├── predict_birds.py   # 鸟类预测脚本
   │       ├── Data/              # 测试数据 (需手动放入 000.unknown 或 具体鸟类文件夹)
   │       └── result/            # [输出] 存放预测报告和热力图
   │
   ├── data/                      # [标准数据] prepare_data.py 生成的训练/验证集
   ├── raw_data/                  # [原始数据] 下载的原始压缩包解压位置
   └── outputs/                   # [实验总输出] 包含权重、日志、训练曲线、KQV分析图
   ```

   ---

   ## 4. 快速开始 (Quick Start)

   ### 4.1 环境搭建
   ```bash
   pip install -r requirements.txt
   ```

   ### 4.2 数据准备

   1. 下载原始数据集，猫狗数据集为实验指导所给的数据集，cub_200数据集是在Kaggle中下载，下载网址为：
   
      ```
      https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images?phase=FinishSSORegistration&returnUrl=%2Fdatasets%2Fveeralakrishna%2F200-bird-species-with-11788-images%2Fversions%2F1%3Fresource%3Ddownload&SSORegistrationToken=CfDJ8MM2ZkWST_dHsDXj-CCe3EOQjiDdX2ZrNrFpDe4eg8PkNBTWVHRKb8bXGy9Las9mqucwBYZ5YMMJ-8G6hc8ZjDendOq4FHn4bKQkVRcFEivoVPaTrpbWZFsp5GbDIjng4DdITn0M5RS2pmw8uUCsLQJpef6DnsTVNvbvsmZVRLtClx7j5wGTTwNSTzZXEv6XTXzhbJgGKlE_XsybccI1nGPYt1_UI5dTJQMYfGVhoU9y8wHPw_zely_4uyoW1Rdy3aiCVyB0OF4tmbapRriqdb3thKIrbzbVE-JzbIRUHh-teH1okqZYeh5wlC-omz-iH7x2WZIApym87P_q1VWs_hTDUg&DisplayName=ling+xuan
      ```
   
   2. 将原始数据集解压至 `raw_data` 后，修改清洗脚本prepare_data中的文件夹名称，与解压文件夹的名称一致。原prepare_data.py中的数据集名称分别为cat_dog与cub_200.

   3. 运行清洗脚本：
   
   ```bash
   python prepare_data.py
   ```
   
   ### 4.3 模型训练 (Training)
   使用 `main.py` 进行训练，程序会自动生成上述提到的所有结果文件。
   
   **任务一：猫狗分类 (ResNet18 + Attention)**
   ```bash
   python main.py --dataset cat_dog --model attention --epochs 5 --batch_size 32
   ```
   
   **任务二：鸟类细粒度分类 (ResNet18 Baseline)**
   ```bash
   python main.py --dataset cub200 --model baseline --epochs 20 --batch_size 16
   ```
   
   **任务三：鸟类细粒度分类 (ResNet18 + Attention)**
   *这是核心实验，将生成 KQV 演变图。*
   ```bash
   python main.py --dataset cub200 --model attention --epochs 20 --batch_size 16 --lr 0.001
   ```
   
   ---
   
   ## 5. Demo 模块使用指南
   
   Demo 模块用于模拟真实场景下的预测，**它会自动对比双模型并在 `result` 目录下生成可视化报告**。
   
   ### 5.1 准备图片
   *   将猫狗测试图放入 `Demo/cat_dog/Data/` 下的 `Cat`, `Dog` 或 `unknown` 文件夹。
   *   将鸟类测试图放入 `Demo/cub200/Data/` 下的 `000.unknown` 或具体类别文件夹。
   
   ### 5.2 运行预测
   ```bash
   # 运行猫狗预测
   cd Demo/cat_dog
   python predict_cat_dog.py
   
   # 运行鸟类预测
   cd ../cub200
   python predict_birds.py
   ```

   ### 5.3 查看结果
   运行结束后，请进入 `Demo/xxx/result/attention/heatmaps/` 查看生成的图片。
   *   **红色区域**代表模型高度关注的区域（Key Factors）。
   *   **蓝色区域**代表模型忽略的背景噪声。
   
   ---
   
   ## 6. 附录：常用指令参数
   
   | 参数           | 默认值     | 说明                                 |
   | :------------- | :--------- | :----------------------------------- |
   | `--dataset`    | `cat_dog`  | 选择数据集 (`cat_dog`, `cub200`)     |
   | `--model`      | `baseline` | 选择模型 (`baseline`, `attention`)   |
   | `--epochs`     | `10`       | 训练轮数                             |
   | `--batch_size` | `32`       | 批处理大小 (Windows遇报错请设为8或4) |
   | `--lr`         | `0.001`    | 初始学习率                           |