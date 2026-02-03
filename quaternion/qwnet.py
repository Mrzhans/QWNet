import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .quaternion_layers import QuaternionConv, QuaternionTransposeConv

class QConv_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=0):
        super(QConv_BN_ACT, self).__init__()
        self.conv = nn.Sequential(
            QuaternionConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=pad, quaternion_format=True),
            nn.BatchNorm2d(out_channels),
            nn.Mish()
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class QTConv_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=0):
        super(QTConv_BN_ACT, self).__init__()
        self.qconv = QuaternionConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=pad, quaternion_format=True)
        self.Tconv = nn.Sequential(
            QuaternionTransposeConv(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=pad, quaternion_format=True),
            nn.BatchNorm2d(in_channels),
            nn.Mish(),
        )
    
    def forward(self, x):
        out = x + self.Tconv(x)
        return self.qconv(out)


# Bidirectional Adaptive Attention Module
class BAAM(nn.Module):
    def __init__(self, channels, num_paths=2, attn_channels=None, act_cfg=dict(type='Mish'), norm_cfg=dict(type='BN', requires_grad=True)):
        super(BAAM, self).__init__()
        self.num_paths = num_paths
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)

        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(attn_channels) if norm_cfg['type'] == 'BN' else nn.GroupNorm(1, attn_channels)
        self.act = nn.ReLU() if act_cfg['type'] == 'ReLU' else nn.Mish()
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

        # Learnable parameters to balance the agg and div features
        self.alpha = nn.Parameter(torch.ones(1))  # For aggregating feature balance
        self.beta = nn.Parameter(torch.ones(1))   # For dividing feature balance

    def forward(self, vi_feature, ir_feature):
        # Aggregate attentions
        x = torch.stack([vi_feature, ir_feature], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = self.sigmoid(attn1)
        attn2 = self.sigmoid(attn2)
        vi_ir_agg = vi_feature * attn1
        ir_vi_agg = ir_feature * attn2

        # Divide attentions
        sub_vi_ir = vi_feature - ir_feature
        attn3 = self.sigmoid(self.avgpool(sub_vi_ir))
        vi_ir_div = sub_vi_ir * attn3

        sub_ir_vi = ir_feature - vi_feature
        attn4 = self.sigmoid(self.avgpool(sub_ir_vi))
        ir_vi_div = sub_ir_vi * attn4
        
        # Apply learnable parameters to balance agg and div features
        vi_output = vi_feature + self.alpha * vi_ir_agg + self.beta * vi_ir_div
        ir_output = ir_feature + self.alpha * ir_vi_agg + self.beta * ir_vi_div

        return vi_output, ir_output


# Wavlet Transform
def create_wavelet_filters():
    """Create Haar wavelet filters."""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) @ harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) @ harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) @ harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) @ harr_wav_H

    filters = {
        'LL': torch.from_numpy(harr_wav_LL).unsqueeze(0),
        'LH': torch.from_numpy(harr_wav_LH).unsqueeze(0),
        'HL': torch.from_numpy(harr_wav_HL).unsqueeze(0),
        'HH': torch.from_numpy(harr_wav_HH).unsqueeze(0)
    }

    return filters

def initialize_wavelet_layer(filter, in_channels, pool=True, device='cuda'):
    """Initialize a wavelet decomposition layer."""
    if pool:
        net = nn.Conv2d(in_channels, in_channels,
                        kernel_size=2, stride=2, padding=0, bias=False,
                        groups=in_channels)
    else:
        net = nn.ConvTranspose2d(in_channels, in_channels,
                                 kernel_size=2, stride=2, padding=0, bias=False,
                                 groups=in_channels)

    net.weight.requires_grad = False
    net.weight.data = filter.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone().to(device)

    return net

def get_wav(in_channels, pool=True):
    """Wavelet decomposition using Conv2d."""
    filters = create_wavelet_filters()

    LL = initialize_wavelet_layer(filters['LL'], in_channels, pool)
    LH = initialize_wavelet_layer(filters['LH'], in_channels, pool)
    HL = initialize_wavelet_layer(filters['HL'], in_channels, pool)
    HH = initialize_wavelet_layer(filters['HH'], in_channels, pool)

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)
        
    def forward(self, x):
        # B C H/2 W/2
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class WaveUnpool(nn.Module):
    def __init__(self, in_channels):
        super(WaveUnpool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels, pool=False)

    def forward(self, ll, lh, hl, hh):
        ll = self.LL(ll)
        lh = self.LH(lh)
        hl = self.HL(hl)
        hh = self.HH(hh)
        return ll + lh + hl + hh


class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Adaptive average pooling
        y = self.avg_pool(x)
        
        # Reshape to apply 1D convolution
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Apply sigmoid activation
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)

class QCMF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=True):
        super(QCMF, self).__init__()
        self.reduceQ = QuaternionConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups, bias=bias, quaternion_format=True)
        self.act = nn.Mish()
        self.qConv = QuaternionConv(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups, bias=bias, quaternion_format=True)
        self.norm = nn.BatchNorm2d(out_channels)
        self.eca = ECALayer(channel=out_channels)
        self.output_conv = QuaternionConv(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        # concat
        fuse = torch.concat([x1, x2], dim=1)
        reduce_fuse = self.reduceQ(fuse)
        # Fusion
        out_level0 = self.act(self.norm(self.qConv(reduce_fuse)))
        # residual
        out_level0 = out_level0 + reduce_fuse
        out_eca = self.eca(out_level0)
        fuse_out = self.output_conv(out_eca)
        return fuse_out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.wav_pool = WavePool(in_channels=1)
        self.vi_conv1 = QConv_BN_ACT(in_channels=4, kernel_size=3, out_channels=8, stride=1, pad=1)
        self.ir_conv1 = QConv_BN_ACT(in_channels=4, kernel_size=3, out_channels=8, stride=1, pad=1)

        self.vi_qconv2 = QConv_BN_ACT(in_channels=8, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_qconv2 = QConv_BN_ACT(in_channels=8, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_qconv3 = QConv_BN_ACT(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_qconv3 = QConv_BN_ACT(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_qconv4 = QConv_BN_ACT(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_qconv4 = QConv_BN_ACT(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.vi_qconv5 = QConv_BN_ACT(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.ir_qconv5 = QConv_BN_ACT(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)

        self.ecaf1 = BAAM(channels=16, num_paths=2, attn_channels=8)
        self.ecaf2 = BAAM(channels=32, num_paths=2, attn_channels=8)
        self.ecaf3 = BAAM(channels=64, num_paths=2, attn_channels=8)

    def forward(self, y_vi_image, ir_image):
        ll_1, lh_1, hl_1, hh_1 = self.wav_pool(y_vi_image)
        ll_2, lh_2, hl_2, hh_2 = self.wav_pool(ir_image)
        vis_Q = torch.cat([ll_1, lh_1, hl_1, hh_1], 1)
        ir_Q = torch.cat([ll_2, lh_2, hl_2, hh_2], 1)
        
        vi_out, ir_out = self.vi_conv1(vis_Q), self.ir_conv1(ir_Q)
        vi_out, ir_out = self.ecaf1(self.vi_qconv2(vi_out), self.ir_qconv2(ir_out))
        vi_out, ir_out = self.ecaf2(self.vi_qconv3(vi_out), self.ir_qconv3(ir_out))
        vi_out, ir_out = self.ecaf3(self.vi_qconv4(vi_out), self.ir_qconv4(ir_out))
        vi_out, ir_out = self.vi_qconv5(vi_out), self.ir_qconv5(ir_out)
        
        return vi_out, ir_out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.wav_t = WaveUnpool(in_channels=1)
        # self.conv1 = QTConv_BN_ACT(in_channels=256, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv1 = QTConv_BN_ACT(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv2 = QTConv_BN_ACT(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv3 = QTConv_BN_ACT(in_channels=32, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.conv4 = QTConv_BN_ACT(in_channels=16, kernel_size=3, out_channels=8, stride=1, pad=1)
        self.conv5 = QTConv_BN_ACT(in_channels=8, kernel_size=3, out_channels=4, stride=1, pad=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        ll, lh, hl, hh = torch.split(x, x.size(1) // 4, dim=1)
        x = self.wav_t(ll, lh, hl, hh)
        # x = F.interpolate(x, size=(inp_img.size(2), inp_img.size(3)), mode='bilinear', align_corners=False)
        x = inp_img + x
        return self.sigmoid(x)


class QWNet(nn.Module):
    def __init__(self):
        super(QWNet, self).__init__()
        self.encoder = Encoder()
        self.qcmf = QCMF(in_channels=256, out_channels=128)
        self.decoder = Decoder()

    def forward(self, y_vi_image, ir_image):
        vi_encoder_out, ir_encoder_out = self.encoder(y_vi_image, ir_image)
        encoder_out = self.qcmf(vi_encoder_out, ir_encoder_out)
        fused_image = self.decoder(y_vi_image, encoder_out)
        return fused_image
    

if __name__ == '__main__':
    from utils.img_read_save import image_read_cv2
    import cv2
    vis_y = torch.FloatTensor(cv2.split(image_read_cv2('/data2/yjt/MMIF-CDDFuse/test_img/M3FD/vis/00390.png', mode='YCrCb'))[0]).unsqueeze(0).to('cuda')
    ir = torch.FloatTensor(image_read_cv2('/data2/yjt/MMIF-CDDFuse/test_img/M3FD/ir/00390.png', mode='GRAY')).unsqueeze(0).to('cuda')
    LL, LH, HL, HH = get_wav(in_channels=1)
    ll, lh, hl, hh = LL(vis_y), LH(vis_y), HL(vis_y), HL(vis_y)
    
    def save_tensor_as_image(tensor, filename):
        tensor = tensor.squeeze(0)
        tensor = tensor.cpu().detach().numpy()
        cv2.imwrite(filename, tensor)
        
    save_tensor_as_image(ll, 'test_result/vis_ll.png')
    save_tensor_as_image(lh, 'test_result/vis_lh.png')
    save_tensor_as_image(hl, 'test_result/vis_hl.png')
    save_tensor_as_image(hh, 'test_result/vis_hh.png')