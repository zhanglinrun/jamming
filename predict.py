import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

# === 配置参数 ===
# 与训练脚本中的 JAM_LABEL_NAMES 保持一致（对应 MATLAB 文件中的函数名）
# J1: 窄带瞄频 -> namj, J2: 宽带阻塞 -> nfmj, J3: 扫频干扰 -> sfj1
# J4: 梳状谱 -> csj, J5: 切片转发 -> isfj, J6: 密集假目标 -> dftj, J7: 窄脉冲 -> npj
JAM_LABEL_NAMES = ['namj',   # J1: 窄带瞄频 (Narrowband Spot)
                   'nfmj',   # J2: 宽带阻塞 (Broadband Barrage)
                   'sfj1',   # J3: 扫频干扰 (Swept)
                   'csj',    # J4: 梳状谱 (Comb)
                   'isfj',   # J5: 切片转发 (Interrupted Sampling)
                   'dftj',   # J6: 密集假目标 (Dense False Target)
                   'npj']    # J7: 窄脉冲 (Narrow Pulse)
NUM_CLASSES = len(JAM_LABEL_NAMES)  # 7类
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5  # 预测阈值

# 默认模型路径（与训练脚本保存的路径一致）
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'radar_multilabel_resnet18.pth')


def load_model(model_path, device):
    """加载预训练模型"""
    # 构建模型结构（与训练时一致）
    model = models.resnet18(weights=None)  # 不使用预训练权重，因为我们要加载自己的权重
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    print(f"✓ 模型已从 {model_path} 加载")
    return model


def get_transform():
    """获取与训练时一致的数据预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def decode_label(vec):
    """将0/1向量解码为干扰类型名称"""
    res = []
    for i, val in enumerate(vec):
        if val == 1:
            res.append(JAM_LABEL_NAMES[i])
    return "+".join(res) if res else "Pure Signal"


def predict_single_image(model, image_path, transform, device, threshold=0.5):
    """预测单张图片"""
    # 读取图片
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"✗ 无法读取图片 {image_path}: {e}")
        return None
    
    # 预处理
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0].cpu().numpy()
        pred_vec = (probs > threshold).astype(int)
    
    return {
        'image_path': image_path,
        'pred_vec': pred_vec,
        'probs': probs,
        'label': decode_label(pred_vec)
    }


def predict_batch(model, image_paths, transform, device, threshold=0.5, batch_size=32):
    """批量预测多张图片"""
    results = []
    
    # 使用 DataLoader 进行批量处理
    from torch.utils.data import Dataset, DataLoader
    
    class ImageDataset(Dataset):
        def __init__(self, image_paths, transform):
            self.image_paths = image_paths
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, img_path
            except Exception as e:
                print(f"✗ 跳过无法读取的图片 {img_path}: {e}")
                return None, img_path
    
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model.eval()
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="预测中", ncols=100):
            # 过滤掉 None（读取失败的图片）
            valid_indices = [i for i, img in enumerate(images) if img is not None]
            if len(valid_indices) == 0:
                continue
            
            valid_images = [images[i] for i in valid_indices]
            valid_paths = [paths[i] for i in valid_indices]
            
            # 转换为 tensor
            input_tensors = torch.stack(valid_images).to(device)
            
            # 批量预测
            outputs = model(input_tensors)
            probs_batch = torch.sigmoid(outputs).cpu().numpy()
            pred_vecs = (probs_batch > threshold).astype(int)
            
            # 保存结果
            for i, img_path in enumerate(valid_paths):
                results.append({
                    'image_path': img_path,
                    'pred_vec': pred_vecs[i],
                    'probs': probs_batch[i],
                    'label': decode_label(pred_vecs[i])
                })
    
    return results


def print_result(result, show_probs=False):
    """打印单个预测结果"""
    print(f"\n图片: {result['image_path']}")
    print(f"预测向量: {result['pred_vec']}")
    if show_probs:
        print(f"预测概率: {result['probs'].round(3)}")
    print(f"干扰类型: {result['label']}")


def main():
    parser = argparse.ArgumentParser(description='雷达干扰多标签预测脚本')
    parser.add_argument('--input', '-i', type=str, default='Comp_J1_Spot_J3_Swept_J4_Comb_J7_NarrowPulse_0001.png',
                        help='输入图片路径或包含图片的文件夹路径')
    parser.add_argument('--model', '-m', type=str, default='radar_multilabel_resnet18.pth',
                        help=f'模型权重文件路径')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='预测阈值 (默认: 0.5)')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='批量预测时的批次大小 (默认: 32)')
    parser.add_argument('--show_probs', '-p', action='store_true',
                        help='显示每个类别的预测概率')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='将结果保存到文件（CSV格式）')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"✗ 错误: 模型文件不存在: {args.model}")
        print(f"  请确保模型文件存在，或使用 --model 参数指定正确的路径")
        return
    
    # 加载模型
    print(f"使用设备: {DEVICE}")
    model = load_model(args.model, DEVICE)
    
    # 获取数据预处理
    transform = get_transform()
    
    # 判断输入是单张图片还是文件夹
    if os.path.isfile(args.input):
        # 单张图片预测
        print(f"\n=== 单张图片预测 ===")
        result = predict_single_image(model, args.input, transform, DEVICE, args.threshold)
        if result:
            print_result(result, args.show_probs)
    elif os.path.isdir(args.input):
        # 文件夹批量预测
        print(f"\n=== 批量预测模式 ===")
        # 查找所有图片文件
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
            image_paths.extend(glob.glob(os.path.join(args.input, ext.upper())))
        
        if len(image_paths) == 0:
            print(f"✗ 在文件夹 {args.input} 中未找到图片文件")
            return
        
        print(f"找到 {len(image_paths)} 张图片")
        
        # 批量预测
        results = predict_batch(model, image_paths, transform, DEVICE, args.threshold, args.batch_size)
        
        # 打印结果
        print(f"\n=== 预测结果 ===")
        for result in results:
            print_result(result, args.show_probs)
        
        # 统计信息
        print(f"\n=== 统计信息 ===")
        label_counts = {}
        for result in results:
            label = result['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"{label}: {count} 张")
        
        # 保存结果到文件
        if args.output:
            import csv
            with open(args.output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['图片路径', '预测向量', '干扰类型', 'J1概率', 'J2概率', 'J3概率', 'J4概率', 'J5概率', 'J6概率', 'J7概率'])
                for result in results:
                    row = [
                        result['image_path'],
                        ''.join(map(str, result['pred_vec'])),
                        result['label']
                    ]
                    row.extend([f"{p:.4f}" for p in result['probs']])
                    writer.writerow(row)
            print(f"\n✓ 结果已保存到: {args.output}")
    else:
        print(f"✗ 错误: 输入路径不存在: {args.input}")
        return


if __name__ == '__main__':
    main()
