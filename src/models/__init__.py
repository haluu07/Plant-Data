from .mobilenet import MobileNetV2Classifier
from .modules import CBAM, MixStyle, AdaptiveConcatPool2d
from .resnet import ResNet50Classifier
from .svm_classifier import SVMModel


def build_model(
    model_name="resnet50",
    num_classes=17,
    pretrained=True,
    dropout=0.2,
    use_cbam=False,
    use_mixstyle=False,
    use_concat_pool=False,
):
    """
    Factory function khởi tạo mô hình học sâu hỗ trợ các kỹ thuật nâng cao.
    """
    model_name = model_name.lower().replace("-", "_")
    if "resnet" in model_name:
        return ResNet50Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            use_cbam=use_cbam,
            use_mixstyle=use_mixstyle,
            use_concat_pool=use_concat_pool,
        )
    elif "mobile" in model_name:
        return MobileNetV2Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            use_cbam=use_cbam,
            use_concat_pool=use_concat_pool,
        )
    else:
        raise ValueError(f"Không hỗ trợ kiến trúc mô hình: {model_name}. Vui lòng chọn 'resnet50' hoặc 'mobilenet_v2'.")


__all__ = [
    "MobileNetV2Classifier",
    "ResNet50Classifier",
    "SVMModel",
    "CBAM",
    "MixStyle",
    "AdaptiveConcatPool2d",
    "build_model",
]
