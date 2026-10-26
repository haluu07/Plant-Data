import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 cho phân loại bệnh cây trồng.
    Kiến trúc nhẹ, phù hợp cho thiết bị di động và biên (Edge devices / IoT).
    """
    def __init__(self, num_classes=17, pretrained=True, dropout=0.2):
        super(MobileNetV2Classifier, self).__init__()
        
        # Tải backbone với trọng số ImageNet
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            self.backbone = mobilenet_v2(weights=weights)
        else:
            self.backbone = mobilenet_v2(weights=None)

        # Thay thế tầng classifier cuối
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)

    def get_last_conv_layer(self):
        """Trả về tầng tích chập cuối cùng để phục vụ Grad-CAM"""
        return self.backbone.features[-1]
