# PyTorch Transforms 使用文档

## 1. 概述

`torchvision.transforms` 提供了一系列图像预处理和数据增强工具，用于在图像输入模型前对其进行变换。这些变换可以组合使用，形成完整的预处理流程。

---

## 2. 常用变换分类

### 2.1 几何变换（空间变换）

| 变换 | 功能 | 常用参数 |
|------|------|----------|
| `Resize(size)` | 调整图像大小 | `size=(H, W)` 或 `size=S`（短边适配） |
| `CenterCrop(size)` | 中心裁剪 | `size=(H, W)` 或 `size=S` |
| `RandomCrop(size)` | 随机裁剪 | `size`, `padding` |
| `RandomResizedCrop(size)` | 随机缩放裁剪 | `size`, `scale=(0.08, 1.0)` |
| `RandomHorizontalFlip(p)` | 随机水平翻转 | `p=0.5`（默认） |
| `RandomVerticalFlip(p)` | 随机垂直翻转 | `p=0.5` |
| `RandomRotation(degrees)` | 随机旋转 | `degrees=30`（[-30, 30]度） |
| `RandomAffine(...)` | 随机仿射变换 | `degrees`, `translate`, `scale`, `shear` |

### 2.2 颜色变换

| 变换 | 功能 | 常用参数 |
|------|------|----------|
| `ColorJitter(...)` | 颜色抖动 | `brightness`, `contrast`, `saturation`, `hue` |
| `Grayscale(num_output_channels)` | 转灰度 | `1` 或 `3` |
| `RandomGrayscale(p)` | 随机转灰度 | `p=0.1` |
| `GaussianBlur(kernel_size, sigma)` | 高斯模糊 | `kernel_size=5`, `sigma=(0.1, 2.0)` |

### 2.3 张量转换与归一化

| 变换 | 功能 | 说明 |
|------|------|------|
| `ToTensor()` | PIL/Image → Tensor | 像素值从 [0,255] 归一化到 [0,1] |
| `Normalize(mean, std)` | 标准化 | `(x - mean) / std`，通常用 ImageNet 统计值 |
| `Pad(padding)` | 边缘填充 | `padding=4` 四边各填充4像素 |

### 2.4 其他实用变换

| 变换 | 功能 |
|------|------|
| `Compose(transforms)` | 组合多个变换 |
| `RandomApply(transforms, p)` | 以概率 p 应用一组变换 |
| `RandomChoice(transforms)` | 随机选择一个变换应用 |

---

## 3. 典型使用场景

### 3.1 训练集数据增强

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize([128, 128]),           # 统一尺寸
    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
    transforms.RandomRotation(30),           # 随机旋转
    transforms.ColorJitter(# 颜色抖动
        brightness=0.3, 
        contrast=0.3, 
        saturation=0.3
    ),                                        
    transforms.RandomAffine(
        degrees=0,  # 随机平移
        translate=(0.1, 0.1)# 转张量
    ),                                       
    transforms.ToTensor(),                    
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )                                         # ImageNet 标准化
])
```

### 3.2 验证/测试集预处理

```python
val_transform = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**注意**：验证集通常不做随机增强（如翻转、旋转），只做必要的尺寸调整和归一化。

### 3.3 测试集预处理

```python
test_transform = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**验证集 vs 测试集的区别**：

| 特性 | 验证集 (Validation) | 测试集 (Test) |
|------|---------------------|---------------|
| **用途** | 调参、选择模型、早停 | 最终评估模型性能 |
| **访问频率** | 训练过程中多次使用 | 只在最后使用一次 |
| **Transforms** | 与测试集相同，无随机增强 | 无随机增强，固定预处理 |
| **数据泄露** | 可能间接影响（调参） | 绝对不能有任何信息泄露 |

**关键原则**：
- 测试集的 transform 必须与验证集**完全一致**
- 测试集绝对不能使用训练集的任何统计信息（如自己计算的 mean/std）
- 如果使用预训练模型，使用模型预训练时的归一化参数

### 3.4 应用到 Dataset

```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 训练集
train_dataset = ImageFolder(
    root='data/train',
    transform=train_transform
)

# 验证集
val_dataset = ImageFolder(
    root='data/val',
    transform=val_transform
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

---

## 4. 参数详解

### 4.1 ColorJitter 参数

```python
transforms.ColorJitter(
    brightness=0.3,    # 亮度变化范围 [1-0.3, 1+0.3] = [0.7, 1.3]
    contrast=0.3,      # 对比度变化范围
    saturation=0.3,    # 饱和度变化范围
    hue=0.1            # 色调变化范围 [-0.1, 0.1]
)
```

### 4.2 RandomAffine 参数

```python
transforms.RandomAffine(
    degrees=30,              # 旋转角度范围
    translate=(0.1, 0.1),    # 平移比例 (x, y)
    scale=(0.9, 1.1),        # 缩放范围
    shear=10                 # 剪切角度
)
```

### 4.3 Normalize 常用值

| 数据集 | mean | std |
|--------|------|-----|
| ImageNet | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] |
| CIFAR-10 | [0.4914, 0.4822, 0.4465] | [0.2470, 0.2435, 0.2616] |
| MNIST | [0.1307] | [0.3081] |

---

## 5. 自定义变换

```python
import random
from PIL import Image

class RandomAddNoise:
    """自定义变换：随机添加噪声"""
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor
    
    def __call__(self, tensor):
        # tensor: [C, H, W]
        noise = torch.randn_like(tensor) * self.noise_factor
        return torch.clamp(tensor + noise, 0, 1)

# 使用自定义变换
transform = transforms.Compose([
    transforms.ToTensor(),
    RandomAddNoise(noise_factor=0.05)
])
```

---

## 6. 注意事项

1. **执行顺序**：几何变换 → 颜色变换 → ToTensor → Normalize
2. **随机性**：随机变换只在 `__call__` 时生效，每次调用结果可能不同
3. **数据类型**：PIL Image 和 Tensor 支持的变换不同，注意转换时机
4. **归一化**：使用预训练模型时，归一化参数必须与预训练时一致

---

## 7. 快速参考

```python
# 基础增强组合
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 强增强组合（适用于小数据集）
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
```
