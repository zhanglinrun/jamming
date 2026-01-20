# 雷达干扰多标签识别系统

## 1. 项目简介 (Introduction)

### 项目目标
本项目是一个基于深度学习的**雷达干扰多标签识别系统**，用于识别和分类雷达信号中的多种干扰类型。系统能够同时识别一张雷达时频图像中存在的多种干扰类型（多标签分类），这对于实际雷达系统中的复合干扰场景具有重要意义。

**核心解决的问题：**
- 识别7种不同类型的雷达干扰（J1-J7）
- 支持复合干扰识别（一张图像中可能同时存在多种干扰）
- 将雷达信号的时频图转换为图像，利用计算机视觉技术进行分类

### 核心技术
- **模型架构**: 基于 ResNet18 的深度卷积神经网络
- **任务类型**: 多标签二分类（Multi-label Binary Classification）
- **损失函数**: BCEWithLogitsLoss（二元交叉熵损失）
- **评估指标**: Exact Match Accuracy（完全匹配准确率）

### 框架
本项目基于 **PyTorch** 深度学习框架实现，使用 `torchvision` 提供的预训练 ResNet18 模型作为特征提取器。

### 干扰类型说明
系统可识别以下7种雷达干扰类型：

| 干扰编号 | 英文名称 | 中文名称 | 代码标识 |
|---------|---------|---------|---------|
| J1 | Narrowband Spot | 窄带瞄频干扰 | namj |
| J2 | Broadband Barrage | 宽带阻塞干扰 | nfmj |
| J3 | Swept Frequency | 扫频干扰 | sfj1 |
| J4 | Comb Spectrum | 梳状谱干扰 | csj |
| J5 | Interrupted Sampling | 切片转发干扰 | isfj |
| J6 | Dense False Target | 密集假目标干扰 | dftj |
| J7 | Narrow Pulse | 窄脉冲干扰 | npj |

---

## 2. 项目框架 (Project Structure)

### 目录树
```
.
├── data/                    # 数据集生成相关文件
│   ├── dataset/            # 生成的数据集存放目录（被.gitignore忽略）
│   └── generate_dataset.py # Python版本的数据集生成脚本
├── data2/                   # 主要数据集目录
│   ├── dataset/            # 训练数据集（包含多个子文件夹，每个文件夹代表一个类别）
│   └── *.m                 # MATLAB版本的数据集生成脚本（可选）
├── train_multilabel.py     # 模型训练脚本（核心文件）
├── predict.py              # 模型预测/推理脚本（核心文件）
├── requirements.txt        # Python依赖包列表
├── README.md               # 本文件
├── .gitignore              # Git忽略文件配置
├── radar_multilabel_resnet18.pth  # 训练好的模型权重文件（训练后生成）
├── train_curves.png        # 训练曲线图（训练后生成）
└── train_log.txt          # 训练日志文件（训练后生成）
```

### 文件功能详解

#### 2.1 `train_multilabel.py` - 训练脚本

**主要作用：**
这是项目的核心训练脚本，用于训练多标签雷达干扰识别模型。

**关键类和函数：**

1. **`RadarMultiLabelDataset` 类**（第39-89行）
   - 自定义PyTorch数据集类，用于加载雷达干扰图像数据
   - `__init__()`: 扫描数据集目录，根据文件夹名称自动解析多标签（例如：`Comp_J1_Spot_J3_Swept_J4_Comb` 会被解析为 `[1,0,1,1,0,0,0]`）
   - `__getitem__()`: 返回图像张量和对应的多标签向量（7维二进制向量）

2. **`build_model()` 函数**（第101-108行）
   - 构建ResNet18模型结构
   - 将预训练模型的最后一层全连接层替换为7输出的线性层（对应7种干扰类型）
   - 支持使用ImageNet预训练权重进行迁移学习

3. **`train_one_epoch()` 函数**（第111-139行）
   - 执行一个训练轮次
   - 使用BCEWithLogitsLoss计算损失
   - 使用Adam优化器更新模型参数
   - 返回平均训练损失

4. **`evaluate()` 函数**（第142-162行）
   - 在验证集上评估模型性能
   - 使用Exact Match Accuracy（完全匹配准确率）作为评估指标
   - 预测时使用sigmoid激活函数和0.5阈值进行二分类

5. **`plot_curves()` 函数**（第174-203行）
   - 绘制训练损失曲线和验证准确率曲线
   - 保存为PNG图片文件

6. **`decode_label()` 函数**（第165-171行）
   - 将二进制标签向量解码为人类可读的干扰类型名称
   - 例如：`[1,0,1,0,0,0,0]` → `"namj+sfj1"`

**与其他文件的交互：**
- 读取 `data2/dataset/` 目录下的图像数据
- 训练完成后保存模型权重到 `radar_multilabel_resnet18.pth`
- 生成训练曲线图 `train_curves.png` 和日志文件 `train_log.txt`

**命令行参数说明：**
- `--data_dir` / `-d`: 数据集根目录（默认：`data2/dataset`）
- `--batch_size` / `-b`: 批大小（默认：128）
- `--epochs` / `-e`: 训练轮数（默认：30）
- `--lr` / `-l`: 学习率（默认：0.0001）
- `--model_out` / `-o`: 模型保存路径（默认：`radar_multilabel_resnet18.pth`）
- `--no_pretrained`: 不使用ImageNet预训练权重
- `--save_curves`: 训练曲线保存路径（默认：`train_curves.png`）
- `--log_file`: 训练日志保存路径（默认：`train_log.txt`）

---

#### 2.2 `predict.py` - 预测脚本

**主要作用：**
用于加载训练好的模型，对单张图片或整个文件夹的图片进行批量预测。

**关键函数：**

1. **`load_model()` 函数**（第30-43行）
   - 加载预训练模型权重
   - 构建与训练时一致的模型结构
   - 将模型设置为评估模式

2. **`get_transform()` 函数**（第46-52行）
   - 返回与训练时一致的数据预处理流程
   - 包括图像缩放、归一化等操作

3. **`predict_single_image()` 函数**（第64-87行）
   - 对单张图片进行预测
   - 返回预测向量、概率值和人类可读的标签

4. **`predict_batch()` 函数**（第90-147行）
   - 批量预测多张图片
   - 使用DataLoader进行高效批量处理
   - 支持进度条显示

5. **`decode_label()` 函数**（第55-61行）
   - 与训练脚本中的函数功能相同，用于解码标签向量

**与其他文件的交互：**
- 加载 `radar_multilabel_resnet18.pth` 模型权重文件
- 支持预测单张图片或整个文件夹的图片
- 可以将预测结果保存为CSV文件

**命令行参数说明：**
- `--input` / `-i`: 输入图片路径或文件夹路径（默认：`Comp_J1_Spot_J3_Swept_J4_Comb_J7_NarrowPulse_0001.png`）
- `--model` / `-m`: 模型权重文件路径（默认：`radar_multilabel_resnet18.pth`）
- `--threshold` / `-t`: 预测阈值（默认：0.5）
- `--batch_size` / `-b`: 批量预测时的批次大小（默认：32）
- `--show_probs` / `-p`: 显示每个类别的预测概率
- `--output` / `-o`: 将结果保存到CSV文件

---

#### 2.3 `data/generate_dataset.py` - 数据集生成脚本

**主要作用：**
用于生成雷达干扰数据集。该脚本模拟雷达信号生成过程，添加不同类型的干扰，然后生成时频图（spectrogram）作为训练数据。

**关键函数：**

1. **信号生成函数**
   - `signal_func()`: 生成基础雷达信号（线性调频信号）

2. **干扰生成函数**（对应7种干扰类型）
   - `namj()`: 窄带瞄频干扰
   - `nfmj()`: 宽带阻塞干扰
   - `sfj1()`: 扫频干扰
   - `csj()`: 梳状谱干扰
   - `isfj()`: 切片转发干扰
   - `dftj()`: 密集假目标干扰
   - `npj()`: 窄脉冲干扰

3. **抗干扰处理函数**
   - `pulse_compression()`: 脉冲压缩
   - `sidelobe_blanking()`: 副瓣匿影
   - `sidelobe_cancellation()`: 副瓣对消
   - `dynamic_anti_jamming()`: 动态抗干扰策略

4. **`main()` 函数**（第524-695行）
   - 主生成逻辑
   - 生成43个类别（1个纯信号 + 7个单干扰 + 35个四种干扰复合）
   - 每个类别生成1000张图片
   - 将时频图保存为PNG格式

**数据集组织结构：**
生成的数据集按照以下规则组织：
- 纯信号：`0_Pure_Signal/`
- 单干扰：`Single_J1_Spot/`, `Single_J2_Barrage/`, ... `Single_J7_NarrowPulse/`
- 复合干扰：`Comp_J1_Spot_J2_Barrage_J3_Swept_J4_Comb/`, ...（共35种组合）

**注意：** 该脚本主要用于生成训练数据。如果已有数据集，可以跳过此步骤。

---

## 3. 环境配置 (Environment Setup)

### 3.1 依赖列表

项目需要以下Python库：

- `torch` >= 1.9.0（PyTorch深度学习框架）
- `torchvision` >= 0.10.0（计算机视觉工具库）
- `numpy` >= 1.21.0（数值计算库）
- `Pillow` >= 8.0.0（图像处理库）
- `matplotlib` >= 3.3.0（绘图库）
- `scipy` >= 1.7.0（科学计算库，用于数据集生成）
- `tqdm` >= 4.62.0（进度条显示库）

### 3.2 安装步骤

#### 方法一：使用 conda（推荐）

```bash
# 1. 创建conda虚拟环境（Python 3.8或更高版本）
conda create -n radar_jamming python=3.8
conda activate radar_jamming

# 2. 安装PyTorch（根据您的CUDA版本选择，如果没有GPU则安装CPU版本）
# 对于CUDA 11.1：
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch

# 对于CPU版本：
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 3. 安装其他依赖
pip install -r requirements.txt
```

#### 方法二：使用 pip 和 venv

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 2. 安装PyTorch（访问 https://pytorch.org/ 获取适合您系统的安装命令）
# 例如，对于CUDA 11.1：
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# 对于CPU版本：
pip install torch torchvision torchaudio

# 3. 安装其他依赖
pip install -r requirements.txt
```

### 3.3 硬件要求

- **CPU**: 建议使用多核CPU（4核以上）
- **内存**: 建议至少8GB RAM
- **GPU**: 可选，但强烈推荐使用NVIDIA GPU（支持CUDA）以加速训练
  - 训练速度：GPU比CPU快10-50倍
  - 支持的CUDA版本：CUDA 10.2, 11.1, 11.3等（根据PyTorch版本而定）
- **存储空间**: 数据集和模型文件需要约几GB到几十GB空间

### 3.4 验证安装

运行以下命令验证环境是否正确配置：

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

如果输出显示CUDA可用，说明GPU环境配置成功。

---

## 4. 使用说明 (Usage)

### 4.1 数据准备 (Data Preparation)

#### 数据集要求

本项目需要的数据集格式如下：

```
data2/dataset/
├── 0_Pure_Signal/
│   ├── 0_Pure_Signal_0001.png
│   ├── 0_Pure_Signal_0002.png
│   └── ...
├── Single_J1_Spot/
│   ├── Single_J1_Spot_0001.png
│   ├── Single_J1_Spot_0002.png
│   └── ...
├── Comp_J1_Spot_J2_Barrage_J3_Swept_J4_Comb/
│   ├── Comp_J1_Spot_J2_Barrage_J3_Swept_J4_Comb_0001.png
│   └── ...
└── ...（其他类别文件夹）
```

**数据集组织规则：**
- 每个子文件夹代表一个类别
- 文件夹名称中包含干扰类型标识（J1, J2, ..., J7）
- 每个文件夹内包含该类别对应的PNG格式图片
- 图片尺寸建议为224x224像素（训练脚本会自动调整）

#### 生成数据集（可选）

如果您需要自己生成数据集，可以使用 `data/generate_dataset.py` 脚本：

```bash
# 进入项目根目录
cd /home2/lrzhang/pythonProject/jamming

# 运行数据集生成脚本
python data/generate_dataset.py
```

**注意：**
- 生成43个类别，每类1000张图片，共约43,000张图片
- 生成过程可能需要较长时间（数小时到数天，取决于硬件性能）
- 生成的数据集将保存在 `data/dataset1/` 目录下
- 您可以在脚本中修改 `dataset_root` 变量来更改保存路径

---

### 4.2 训练流程 (Training Process)

#### 基本训练命令

```bash
# 使用默认参数训练
python train_multilabel.py
```

#### 自定义参数训练

```bash
# 指定数据集路径、批大小、训练轮数等
python train_multilabel.py \
    --data_dir data2/dataset \
    --batch_size 128 \
    --epochs 30 \
    --lr 0.0001 \
    --model_out my_model.pth \
    --save_curves my_curves.png \
    --log_file my_log.txt
```

#### 参数详解

- `--data_dir` / `-d`: 指定数据集根目录
  - 默认值：`data2/dataset`
  - 示例：`--data_dir /path/to/your/dataset`

- `--batch_size` / `-b`: 批大小（每次训练使用的样本数）
  - 默认值：128
  - 建议值：32-256（根据GPU显存调整，显存越大可以设置越大）
  - 示例：`--batch_size 64`

- `--epochs` / `-e`: 训练轮数
  - 默认值：30
  - 建议值：20-50（根据数据集大小和收敛情况调整）
  - 示例：`--epochs 50`

- `--lr` / `-l`: 学习率
  - 默认值：0.0001
  - 建议范围：0.00001 - 0.001
  - 如果训练损失不下降，可以尝试降低学习率
  - 示例：`--lr 0.0005`

- `--model_out` / `-o`: 模型保存路径
  - 默认值：`radar_multilabel_resnet18.pth`
  - 示例：`--model_out models/best_model.pth`

- `--no_pretrained`: 不使用ImageNet预训练权重
  - 默认：使用预训练权重（推荐，可以加快收敛）
  - 如果添加此参数，模型将从头开始训练

- `--save_curves`: 训练曲线保存路径
  - 默认值：`train_curves.png`
  - 示例：`--save_curves results/training_curves.png`

- `--log_file`: 训练日志保存路径
  - 默认值：`train_log.txt`
  - 示例：`--log_file logs/train_20231201.txt`

#### 训练过程说明

1. **数据加载阶段**
   - 脚本会扫描数据集目录，解析每个文件夹的多标签
   - 自动将数据集划分为训练集（80%）和验证集（20%）

2. **训练阶段**
   - 每个epoch会显示训练进度条和当前损失值
   - 每个epoch结束后会在验证集上评估模型性能
   - 训练日志会同时输出到终端和日志文件

3. **训练输出**
   - 模型权重文件：`radar_multilabel_resnet18.pth`（或您指定的路径）
   - 训练曲线图：`train_curves.png`（包含训练损失和验证准确率曲线）
   - 训练日志：`train_log.txt`（包含详细的训练过程记录）

#### 训练示例输出

```
使用设备: cuda
数据目录: data2/dataset
batch_size=128, epochs=30, lr=0.0001
正在扫描数据集并解析多标签...
解析完成! 共找到 43000 张图片。

Epoch 1/30
----------
Epoch 1/30 [Train]: 100%|████████| 269/269 [00:45<00:00, loss=0.5234]
Train Loss: 0.5234
Val Exact Match Acc: 0.6234

Epoch 2/30
----------
...
```

---

### 4.3 测试/推理流程 (Testing/Inference Process)

#### 单张图片预测

```bash
# 预测单张图片
python predict.py --input Comp_J1_Spot_J3_Swept_J4_Comb_J7_NarrowPulse_0001.png
```

#### 批量预测（整个文件夹）

```bash
# 预测整个文件夹中的所有图片
python predict.py --input data2/dataset/Comp_J1_Spot_J2_Barrage_J3_Swept_J4_Comb/ --batch_size 64
```

#### 完整参数示例

```bash
# 使用自定义模型、显示概率、保存结果到CSV
python predict.py \
    --input data2/dataset/test_images/ \
    --model radar_multilabel_resnet18.pth \
    --threshold 0.5 \
    --batch_size 32 \
    --show_probs \
    --output predictions.csv
```

#### 参数详解

- `--input` / `-i`: 输入图片路径或文件夹路径
  - 可以是单张图片：`--input image.png`
  - 也可以是文件夹：`--input /path/to/folder/`
  - 默认值：`Comp_J1_Spot_J3_Swept_J4_Comb_J7_NarrowPulse_0001.png`

- `--model` / `-m`: 模型权重文件路径
  - 默认值：`radar_multilabel_resnet18.pth`
  - 确保该文件存在，否则会报错
  - 示例：`--model models/best_model.pth`

- `--threshold` / `-t`: 预测阈值
  - 默认值：0.5
  - 当某个类别的预测概率大于此阈值时，认为该类别存在
  - 可以调整此值来平衡精确率和召回率
  - 示例：`--threshold 0.6`（更严格，减少误报）

- `--batch_size` / `-b`: 批量预测时的批次大小
  - 默认值：32
  - 批量预测时使用，可以加快预测速度
  - 示例：`--batch_size 64`

- `--show_probs` / `-p`: 显示每个类别的预测概率
  - 添加此参数会显示每个干扰类型的详细概率值
  - 不加此参数只显示最终的预测结果

- `--output` / `-o`: 将结果保存到CSV文件
  - 指定CSV文件路径，预测结果会保存到该文件
  - CSV文件包含：图片路径、预测向量、干扰类型、各类别概率
  - 示例：`--output results/predictions.csv`

#### 预测输出示例

**单张图片预测输出：**
```
使用设备: cuda
✓ 模型已从 radar_multilabel_resnet18.pth 加载

=== 单张图片预测 ===

图片: Comp_J1_Spot_J3_Swept_J4_Comb_J7_NarrowPulse_0001.png
预测向量: [1 0 1 1 0 0 1]
预测概率: [0.95 0.12 0.87 0.91 0.08 0.05 0.78]
干扰类型: namj+sfj1+csj+npj
```

**批量预测输出：**
```
使用设备: cuda
✓ 模型已从 radar_multilabel_resnet18.pth 加载

=== 批量预测模式 ===
找到 1000 张图片
预测中: 100%|████████| 32/32 [00:15<00:00]

=== 预测结果 ===
（显示每张图片的预测结果）

=== 统计信息 ===
namj+sfj1+csj+npj: 450 张
namj+sfj1+csj: 320 张
Pure Signal: 230 张
...

✓ 结果已保存到: predictions.csv
```

#### 结果文件格式（CSV）

如果使用 `--output` 参数，生成的CSV文件格式如下：

| 图片路径 | 预测向量 | 干扰类型 | J1概率 | J2概率 | J3概率 | J4概率 | J5概率 | J6概率 | J7概率 |
|---------|---------|---------|-------|-------|-------|-------|-------|-------|-------|
| image1.png | 1011000 | namj+sfj1 | 0.9500 | 0.1200 | 0.8700 | 0.0500 | 0.0800 | 0.0300 | 0.0200 |
| image2.png | 0000000 | Pure Signal | 0.0500 | 0.0300 | 0.0200 | 0.0100 | 0.0100 | 0.0100 | 0.0100 |

---

## 5. 总结 (Conclusion)

### 项目特点

本项目实现了一个完整的雷达干扰多标签识别系统，具有以下特点：

1. **多标签分类能力**：能够同时识别一张图像中的多种干扰类型，更符合实际应用场景
2. **端到端流程**：从数据集生成、模型训练到推理预测的完整流程
3. **易于使用**：提供详细的命令行参数和清晰的输出信息
4. **可扩展性**：基于PyTorch框架，易于修改和扩展

### 未来优化方向

1. **模型改进**
   - 尝试更深的网络（ResNet50, ResNet101）或更先进的架构（EfficientNet, Vision Transformer）
   - 使用注意力机制提升模型性能
   - 尝试集成学习方法

2. **数据处理**
   - 数据增强（旋转、翻转、噪声添加等）
   - 类别不平衡处理（如果某些干扰类型样本较少）
   - 更精细的数据预处理策略

3. **训练策略**
   - 学习率调度（如CosineAnnealingLR）
   - 早停（Early Stopping）机制
   - 模型检查点保存（保存最佳模型）

4. **评估指标**
   - 除了Exact Match Accuracy，还可以添加：
     - Hamming Loss（汉明损失）
     - F1 Score（每个类别的F1分数）
     - Precision/Recall（精确率和召回率）

5. **部署优化**
   - 模型量化（减少模型大小，加快推理速度）
   - ONNX格式导出（便于跨平台部署）
   - 实时推理优化

### 常见问题

**Q: 训练时显存不足怎么办？**
A: 减小 `--batch_size` 参数（如改为32或64），或者使用梯度累积技术。

**Q: 训练损失不下降怎么办？**
A: 尝试降低学习率（如 `--lr 0.00001`），检查数据是否正确加载，或者尝试不使用预训练权重。

**Q: 预测结果不准确怎么办？**
A: 检查模型是否训练充分（增加训练轮数），调整预测阈值，或者增加训练数据量。

**Q: 如何添加新的干扰类型？**
A: 需要修改代码中的 `JAM_FOLDER_KEYWORDS` 和 `JAM_LABEL_NAMES` 列表，并重新训练模型。

---

## 6. 许可证与致谢

本项目仅供学习和研究使用。如有问题或建议，欢迎提出Issue或Pull Request。

---

**祝您使用愉快！** 🚀
