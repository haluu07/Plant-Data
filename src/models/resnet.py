import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from .modules import CBAM, MixStyle, AdaptiveConcatPool2d


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 tối ưu hóa cho nhận dạng bệnh cây trồng và Cross-Domain Generalization:
    - Hỗ trợ CBAM Attention (tập trung vết bệnh, triệt tiêu nhiễu nền)
    - Hỗ trợ MixStyle (Domain Generalization - ICLR 2021)
    - Hỗ trợ Adaptive Concat Pooling (bắt cả ngữ cảnh và đốm bệnh nhỏ)
    """
    def __init__(
        self,
        num_classes=17,
        pretrained=True,
        dropout=0.3,
        use_cbam=False,
        use_mixstyle=False,
        use_concat_pool=False,
    ):
        super(ResNet50Classifier, self).__init__()
        self.use_cbam = use_cbam
        self.use_mixstyle = use_mixstyle
        self.use_concat_pool = use_concat_pool
        self.num_classes = num_classes

        # Tải backbone với trọng số ImageNet
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)

        # MixStyle tích hợp sau layer1 và layer2 để trộn style màu sắc/nền
        if self.use_mixstyle:
            self.mixstyle = MixStyle(p=0.5, alpha=0.1)

        # CBAM Attention Module tại tầng đặc trưng sâu nhất (2048 channels)
        if self.use_cbam:
            self.cbam = CBAM(in_planes=2048, ratio=16)

        # Head phân loại
        in_features = self.backbone.fc.in_features
        if self.use_concat_pool:
            self.pool = AdaptiveConcatPool2d(1)
            head_in = in_features * 2
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            head_in = in_features

        # Classifier head cải tiến với BatchNorm1d và GELU
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Dropout(p=dropout),
            nn.Linear(head_in, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(512, num_classes),
        )

        # Gỡ bỏ fc mặc định của torchvision
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()

    def forward(self, x):
        # Layer stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Stage 1
        x = self.backbone.layer1(x)
        if self.use_mixstyle:
            x = self.mixstyle(x)

        # Stage 2
        x = self.backbone.layer2(x)
        if self.use_mixstyle:
            x = self.mixstyle(x)

        # Stage 3 & 4
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Tích hợp CBAM nếu được kích hoạt
        if self.use_cbam:
            x = self.cbam(x)

        # Pooling và Classifier
        feat = self.pool(x)
        feat = torch.flatten(feat, 1)
        out = self.classifier(feat)
        return out

    def get_last_conv_layer(self):
        """Trả về tầng tích chập cuối cùng để phục vụ Grad-CAM"""
        return self.backbone.layer4[-1]

    def get_parameter_groups(self, base_lr, backbone_lr_ratio=0.1):
        """
        Tách các nhóm tham số để áp dụng Differential Learning Rate:
        - Backbone: lr nhỏ để giữ đặc trưng ImageNet
        - Attention & Classifier Head: lr chuẩn để học nhanh đặc trưng bệnh
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
