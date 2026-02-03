import torch
import cv2
import numpy as np
from sklearn.manifold import TSNE
from utils.img_read_save import image_read_cv2
from quaternion.qwnet import get_wav
import matplotlib.pyplot as plt


def load_wavelet_components(vis_y_path, ir_path):
    vis_y = torch.FloatTensor(cv2.split(image_read_cv2(vis_y_path, mode='YCrCb'))[0]).unsqueeze(0).to('cuda')
    ir = torch.FloatTensor(image_read_cv2(ir_path, mode='GRAY')).unsqueeze(0).to('cuda')
    
    LL, LH, HL, HH = get_wav(in_channels=1)
    vis_components = [LL(vis_y), LH(vis_y), HL(vis_y), HH(vis_y)]
    ir_components = [LL(ir), LH(ir), HL(ir), HH(ir)]
    
    return vis_components, ir_components

def prepare_features(components):
    feature_vectors = []
    for comp in components:
        comp_np = comp.squeeze(0).cpu().detach().numpy().flatten()
        feature_vectors.append(comp_np)
    
    return np.array(feature_vectors)

def visualize_tsne(vis_features, ir_features):
    features = np.vstack([vis_features, ir_features]).astype(np.float32)
    tsne = TSNE(n_components=2, random_state=0, perplexity=2)
    transformed = tsne.fit_transform(features)

    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed[:4, 0], transformed[:4, 1], c='blue', label='Visible', marker='o')
    plt.scatter(transformed[4:, 0], transformed[4:, 1], c='red', label='Infrared', marker='x')
    plt.legend()
    plt.title('t-SNE Visualization of Wavelet Components')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()  # 显示图像
    
    plt.savefig('tsne_visualization.jpg', format='jpg', dpi=300)  # 然后保存


# 读取图像并处理
vis_y_path = '/data2/yjt/Datasets/M3FD/test/vis/00390.png'
ir_path = '/data2/yjt/Datasets/M3FD/test/ir/00390.png'
vis_components, ir_components = load_wavelet_components(vis_y_path, ir_path)

# 准备特征向量
vis_features = prepare_features(vis_components)
ir_features = prepare_features(ir_components)

# t-SNE 可视化
visualize_tsne(vis_features, ir_features)
