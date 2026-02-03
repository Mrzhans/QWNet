# QWNet

QWNet: A Quaternion Wavelet Network for Spatial-Frequency Aware Multi-modal Image Fusion (Neural Networks 2025)

## Introduction（简介）

QWNet is a deep learning project for multi-modal image fusion using quaternion convolutional neural networks. The network can effectively fuse infrared (IR) and visible (VIS) images, and can also be used for medical image fusion (MIF) tasks.

（QWNet 是一个利用四元数卷积神经网络进行多模态图像融合的深度学习网络。该网络能够有效地融合红外（IR）和可见光（VIS）图像，也可用于医学图像融合（MIF）任务。）

## Project Structure（项目结构）

```
QWNet/
├── quaternion/              # Quaternion neural network modules（四元数神经网络模块）
│   ├── __init__.py
│   ├── quaternion_layers.py # Quaternion convolution layers（四元数卷积层实现）
│   ├── quaternion_ops.py    # Quaternion operations（四元数操作）
│   └── qwnet.py             # QWNet architecture（QWNet 网络架构）
├── utils/                   # Utility functions（工具函数）
│   ├── dataset.py           # Dataset loading（数据集加载）
│   ├── Evaluator.py         # Evaluation metrics（评估指标）
│   ├── img_read_save.py     # Image I/O（图像读写）
│   ├── loss.py              # Loss functions（损失函数）
│   ├── loss_ssim.py         # SSIM loss（SSIM 损失）
│   └── loss_vif.py          # VIF loss（VIF 损失）
├── train_IVF.py             # Infrared-visible fusion training（红外可见光图像融合训练脚本）
├── train_MIF.py             # Medical image fusion training（医学图像融合训练脚本）
├── test.py                  # Testing script（测试脚本）
├── analysis_t-SNE.py        # t-SNE feature visualization（t-SNE 特征可视化分析）
├── dataprocessing.py        # Data preprocessing（数据预处理脚本）
└── requirements.txt         # Dependencies list（依赖包列表）
```

## Requirements（环境要求）

- Python 3.9+
- PyTorch 1.13.1+

## Installation（安装）

1. Clone the repository（克隆仓库）：
```bash
git clone https://github.com/yourusername/QWNet.git
cd QWNet
```

2. Install dependencies（安装依赖）：
```bash
pip install -r requirements.txt
```

## Usage（使用方法）

### Training（训练）

**Infrared and Visible Image Fusion Training（红外与可见光图像融合训练）：**
```bash
python train_IVF.py
```

**Medical Image Fusion Training（医学图像融合训练）：**
```bash
python train_MIF.py
```

### Testing（测试）

```bash
python test.py
```

Results will be saved in the `test_result/` directory.（测试结果将保存在 `test_result/` 目录下。）

## Evaluation Metrics（评估指标）

The following evaluation metrics are used to measure fusion image quality:（本项目使用以下评估指标来衡量融合图像质量：）

| Metric（指标） | Description（描述） |
|------|------|
| EN | Entropy（熵） |
| SD | Standard Deviation（标准差） |
| SF | Spatial Frequency（空间频率） |
| MI | Mutual Information（互信息） |
| SCD | Sum of Correlations of Differences（差异相关性之和） |
| SSIM | Structural Similarity（结构相似性） |

## Key Features（主要特性）

- 🔷 **Quaternion Convolution（四元数卷积）**: Leverages quaternion representation for multi-channel image processing（利用四元数表示处理多通道图像信息）
- 🌊 **Wavelet Transform（小波变换）**: Combines wavelet transform for multi-scale feature extraction（结合小波变换进行多尺度特征提取）
- 🎯 **Bidirectional Adaptive Attention Module (BAAM)（双向自适应注意力模块）**: Effectively fuses features from different modalities（有效融合不同模态特征）
- 📊 **Multi-Loss Function（多损失函数）**: Optimizes with gradient loss, L1 loss, and SSIM loss（结合梯度损失、L1损失和SSIM损失进行优化）

## Citation（引用）

If you use this code, please cite our work:（如果您使用了本项目的代码，请引用我们的工作：）

```bibtex
@article{yang2025qwnet,
  title={QWNet: A quaternion wavelet network for spatial-frequency aware multi-modal image fusion},
  author={Yang, Jietao and Lin, Miaoshan and Huang, Guoheng and Chen, Xuhang and Zhang, Xiaofeng and Yuan, Xiaochen and Pun, Chi-Man and Ling, Bingo Wing-Kuen},
  journal={Neural Networks},
  pages={108364},
  year={2025},
  publisher={Elsevier}
}
```

## Contact（联系方式）

For any questions, please open an Issue or contact the authors.（如有问题，请提交 Issue 或联系作者。）
