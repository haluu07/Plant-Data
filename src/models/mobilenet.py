import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from .modules import CBAM, AdaptiveConcatPool2d


class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 tối ưu hóa cho nhận dạng bệnh cây trồng trên Edge/IoT:
    - Trọng số ImageNet khởi tạo
    - Tùy chọn CBAM Attention tại tầng đặc trưng cuối (1280 channels)
    - Head phân loại tối ưu với BatchNorm1d và GELU
    """
    def __init__(
        self,
        num_classes=17,
        pretrained=True,
        dropout=0.2,
        use_cbam=False,
        use_concat_pool=False,
    ):
        super(MobileNetV2Classifier, self).__init__()
        self.num_classes = num_classes
        self.use_cbam = use_cbam
        self.use_concat_pool = use_concat_pool

        # Tải backbone với trọng số ImageNet
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            self.backbone = mobilenet_v2(weights=weights)
        else:
            self.backbone = mobilenet_v2(weights=None)

        in_channels = 1280
        if self.use_cbam:
            self.cbam = CBAM(in_planes=in_channels, ratio=16)

        if self.use_concat_pool:
            self.pool = AdaptiveConcatPool2d(1)
            head_in = in_channels * 2
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            head_in = in_channels

        # Classifier head cải tiến
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Dropout(p=dropout),
            nn.Linear(head_in, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(384, num_classes),
        )

        # Gỡ bỏ classifier mặc định của torchvision
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        feat = self.backbone.features(x)
        if self.use_cbam:
            feat = self.cbam(feat)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        out = self.classifier(feat)
        return out

    def get_last_conv_layer(self):
        """Trả về tầng tích chập cuối cùng để phục vụ Grad-CAM"""
        return self.backbone.features[-1]

    def get_parameter_groups(self, base_lr, backbone_lr_ratio=0.1):
        """
        Tách các nhóm tham số để áp dụng Differential Learning Rate
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name or "cbam" in name or "pool" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_ratio},
            {"params": head_params, "lr": base_lr},
        ]
