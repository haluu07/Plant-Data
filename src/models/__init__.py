from .mobilenet import MobileNetV2Classifier
from .resnet import ResNet50Classifier
from .svm_classifier import SVMModel


def build_model(model_name="resnet50", num_classes=17, pretrained=True, dropout=0.2):
    """
    Factory function khởi tạo mô hình học sâu.
    """
    model_name = model_name.lower().replace("-", "_")
    if "resnet" in model_name:
        return ResNet50Classifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif "mobile" in model_name:
        return MobileNetV2Classifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Không hỗ trợ kiến trúc mô hình: {model_name}. Vui lòng chọn 'resnet50' hoặc 'mobilenet_v2'.")


__all__ = ["MobileNetV2Classifier", "ResNet50Classifier", "SVMModel", "build_model"]
