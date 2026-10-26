import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 cho phân loại bệnh cây trồng.
    Mạng tích chập sâu với Skip-connections làm chuẩn đánh giá mạnh mẽ.
    """
    def __init__(self, num_classes=17, pretrained=True, dropout=0.3):
        super(ResNet50Classifier, self).__init__()
        
        # Tải backbone với trọng số ImageNet
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)

        # Thay thế tầng Fully Connected cuối
        in_features = self.backbone.fc.in_features
        if dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)

    def get_last_conv_layer(self):
        """Trả về tầng tích chập cuối cùng để phục vụ Grad-CAM"""
        return self.backbone.layer4[-1]
