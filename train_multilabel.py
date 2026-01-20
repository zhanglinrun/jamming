import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# === 1. 默认配置参数（可通过命令行覆盖） ===
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data2', 'dataset')
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 30          # 多标签任务通常比单标签需要多练几轮
DEFAULT_LR = 0.0001          # 学习率稍微调小一点，更稳
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义干扰列表 (对应输出向量的顺序)
# 文件夹中使用的关键字: 0->J1, 1->J2, ..., 6->J7
JAM_FOLDER_KEYWORDS = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
# 对应的人类可读干扰名称（与 MATLAB 文件 run_composite_jamming.m 中的函数名对应）
# J1: 窄带瞄频 -> namj, J2: 宽带阻塞 -> nfmj, J3: 扫频干扰 -> sfj1
# J4: 梳状谱 -> csj, J5: 切片转发 -> isfj, J6: 密集假目标 -> dftj, J7: 窄脉冲 -> npj
JAM_LABEL_NAMES = ['namj',   # J1: 窄带瞄频 (Narrowband Spot)
                   'nfmj',   # J2: 宽带阻塞 (Broadband Barrage)
                   'sfj1',   # J3: 扫频干扰 (Swept)
                   'csj',    # J4: 梳状谱 (Comb)
                   'isfj',   # J5: 切片转发 (Interrupted Sampling)
                   'dftj',   # J6: 密集假目标 (Dense False Target)
                   'npj']    # J7: 窄脉冲 (Narrow Pulse)

NUM_CLASSES = len(JAM_FOLDER_KEYWORDS) # 7类

# === 2. 自定义数据集 (核心修改部分) ===
class RadarMultiLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 遍历所有子文件夹
        print("正在扫描数据集并解析多标签...")
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # --- 核心逻辑：根据文件夹名解析出 [0,1,0,1...] ---
            # 初始化全0向量 [0, 0, 0, 0, 0, 0, 0]
            label_vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)

            # 如果是纯信号 '0_Pure_Signal'，label_vec 保持全0
            # 如果是干扰，检查文件名里包含哪些 'J1', 'J2'...
            for idx, key in enumerate(JAM_FOLDER_KEYWORDS):
                if key in folder_name:
                    label_vec[idx] = 1.0

            # 获取该文件夹下所有图片
            img_files = glob.glob(os.path.join(folder_path, "*.png"))
            for img_path in img_files:
                self.image_paths.append(img_path)
                # 使用 clone() 避免潜在引用问题（更安全）
                self.labels.append(label_vec.clone())

        print(f"解析完成! 共找到 {len(self.image_paths)} 张图片。")
        # 打印几个样本看看对不对
        if len(self.labels) > 0:
            sample_idx = min(100, len(self.labels) - 1)
            print(f"示例标签 (对应文件夹 {os.path.basename(os.path.dirname(self.image_paths[sample_idx]))}): {self.labels[sample_idx]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 读取图片并转RGB
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """与预测脚本保持一致的数据预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def build_model(pretrained=True):
    """构建模型结构"""
    from torchvision.models import ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """训练一个 epoch，返回平均 loss"""
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)

    train_pbar = tqdm(
        enumerate(train_loader),
        total=total_batches,
        desc=f'Epoch {epoch}/{total_epochs} [Train]',
        ncols=100
    )

    for batch_idx, (inputs, labels) in train_pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        current_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
        train_pbar.set_postfix(loss=f'{current_loss:.4f}')

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate(model, data_loader, device):
    """在验证集上评估 Exact Match Accuracy"""
    model.eval()
    correct_samples = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            matches = (preds == labels).all(dim=1)
            correct_samples += matches.sum().item()
            total_samples += inputs.size(0)

    val_acc = correct_samples / total_samples if total_samples > 0 else 0.0
    return val_acc


def decode_label(vec):
    """将0/1向量解码为干扰类型名称"""
    res = []
    for i, val in enumerate(vec):
        if val == 1:
            res.append(JAM_LABEL_NAMES[i])
    return "+".join(res) if res else "Pure Signal"


def plot_curves(train_loss_history, val_acc_history, save_path=None):
    """Plot training loss and validation accuracy curves and save to file."""
    epochs = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(10, 4))

    # Subplot 1: Training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)

    # Subplot 2: Validation accuracy (Exact Match)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Exact Match Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.grid(True)

    plt.tight_layout()

    # 默认保存到 train_curves.png，如果未指定路径
    if save_path is None:
        save_path = 'train_curves.png'

    plt.savefig(save_path, dpi=300)
    print(f"训练曲线已保存到: {save_path}")


def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='雷达干扰多标签训练脚本')
    parser.add_argument('--data_dir', '-d', type=str, default=DEFAULT_DATA_DIR,
                        help=f'数据集根目录 (默认: {DEFAULT_DATA_DIR})')
    parser.add_argument('--batch_size', '-b', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'批大小 (默认: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--epochs', '-e', type=int, default=DEFAULT_EPOCHS,
                        help=f'训练轮数 (默认: {DEFAULT_EPOCHS})')
    parser.add_argument('--lr', '-l', type=float, default=DEFAULT_LR,
                        help=f'学习率 (默认: {DEFAULT_LR})')
    parser.add_argument('--model_out', '-o', type=str, default='radar_multilabel_resnet18.pth',
                        help='模型保存路径 (默认: radar_multilabel_resnet18.pth)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='不使用 ImageNet 预训练权重')
    parser.add_argument('--save_curves', type=str, default='train_curves.png',
                        help='将训练曲线保存为图片的路径 (默认: train_curves.png)')
    parser.add_argument('--log_file', type=str, default='train_log.txt',
                        help='训练日志保存路径 (默认: train_log.txt)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir = args.data_dir
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    model_out = args.model_out
    log_file = args.log_file

    # === 日志初始化 ===
    def log(msg: str):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    # 清空旧日志
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== 训练日志开始 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    log(f"使用设备: {DEVICE}")
    log(f"数据目录: {data_dir}")
    log(f"batch_size={batch_size}, epochs={epochs}, lr={lr}")

    # === 数据预处理 ===
    data_transforms = get_transforms()

    # 加载数据集
    full_dataset = RadarMultiLabelDataset(data_dir, transform=data_transforms)

    # 划分训练/测试集
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # === 构建模型 ===
    model = build_model(pretrained=not args.no_pretrained).to(DEVICE)

    # 损失和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    train_loss_history = []
    val_acc_history = []

    for epoch in range(1, epochs + 1):
        log(f'\nEpoch {epoch}/{epochs}')
        log('-' * 10)

        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, epochs)
        train_loss_history.append(epoch_loss)
        log(f'Train Loss: {epoch_loss:.4f}')

        val_acc = evaluate(model, test_loader, DEVICE)
        val_acc_history.append(val_acc)
        log(f'Val Exact Match Acc: {val_acc:.4f}')

    # 保存模型
    torch.save(model.state_dict(), model_out)
    log(f"\n模型已保存到: {model_out}")

    # 随机测试一张图
    log("\n=== 随机测试演示 ===")
    model.eval()
    idx = torch.randint(0, len(test_dataset), (1,)).item()
    img, true_label = test_dataset[idx]

    with torch.no_grad():
        input_tensor = img.unsqueeze(0).to(DEVICE)
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0]
        pred_label = (probs > 0.5).float()

    log(f"真实标签: {true_label.cpu().numpy().astype(int)}")
    log(f"预测标签: {pred_label.cpu().numpy().astype(int)}")
    log(f"预测概率: {probs.cpu().numpy().round(2)}")

    log(f"解释: 真实[{decode_label(true_label)}] vs 预测[{decode_label(pred_label)}]")

    # 绘制训练曲线
    plot_curves(train_loss_history, val_acc_history, save_path=args.save_curves)


if __name__ == '__main__':
    main()