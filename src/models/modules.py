import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CBAM):
    Học trọng số cho từng kênh đặc trưng, giúp mạng tập trung vào 'bệnh gì' (triệu chứng nấm/vi khuẩn).
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(in_planes // ratio, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (CBAM):
    Học trọng số không gian (vị trí pixel), giúp triệt tiêu bối cảnh nền (đất, cỏ, phông studio)
    và ép mạng chỉ chú ý vào vị trí các vết đốm bệnh trên phiến lá.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) kết hợp Channel & Spatial Attention.
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_planes, ratio=ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class AdaptiveConcatPool2d(nn.Module):
    """
    Kết hợp cả AdaptiveAvgPool2d (bắt ngữ cảnh tổng thể) và AdaptiveMaxPool2d (bắt đốm bệnh nhỏ nhất).
    Tạo vector đặc trưng giàu thông tin gấp đôi so với AvgPool thông thường.
    """
    def __init__(self, size=1):
        super(AdaptiveConcatPool2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(size)
        self.max_pool = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)


class MixStyle(nn.Module):
    """
    MixStyle (ICLR 2021) - Kỹ thuật Domain Generalization hàng đầu:
    Trộn thống kê đặc trưng (mean và standard deviation) giữa các ảnh trong batch trong lúc train.
    Làm cho mạng trở nên 'bất biến' với phong cách ảnh (màu nền phòng lab vs ánh sáng ngoài vườn).
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super(MixStyle, self).__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)
        # Tính mean và std theo từng instance và channel
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()

        # Chuẩn hóa đặc trưng
        x_normed = (x - mu) / sig

        # Hoán vị ngẫu nhiên thứ tự các mẫu trong batch
        perm = torch.randperm(B)

        # Lấy tham số trộn từ phân phối Beta
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix
