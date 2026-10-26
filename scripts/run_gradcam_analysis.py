import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.datasets.transforms import get_transforms
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.xai.gradcam import GradCAM, overlay_cam_on_image


def analyze_and_plot(model, gradcam, img_path_lab, img_path_field, class_names, device, save_path):
    """
    Sinh biểu đồ so sánh Grad-CAM giữa ảnh Lab (PlantVillage) và ảnh Thực địa (PlantDoc).
    Làm nổi bật sự khác biệt về vùng chú ý của mạng (Focus on lesion vs Background noise).
    """
    transform = get_transforms(image_size=(224, 224), is_train=False)

    pairs = [
        ("Ảnh Phòng Lab (PlantVillage)", img_path_lab),
        ("Ảnh Thực Địa (PlantDoc)", img_path_field),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.suptitle("Phân tích Hiện tượng Shortcut Learning & Background Bias bằng Grad-CAM", fontsize=16, fontweight="bold", y=0.98)

    for row_idx, (domain_title, img_path) in enumerate(pairs):
        if not Path(img_path).exists():
            print(f"[-] Không tìm thấy ảnh: {img_path}")
            continue

        raw_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(raw_img).unsqueeze(0).to(device)

        # Forward pass để lấy prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            conf = probs[pred_idx].item() * 100

        pred_name = class_names[pred_idx] if class_names and pred_idx < len(class_names) else f"Class {pred_idx}"

        # Sinh Grad-CAM heatmap
        cam, _ = gradcam.generate(input_tensor, target_class=pred_idx)
        overlay_img, heatmap_colored = overlay_cam_on_image(raw_img, cam, alpha=0.55)

        # Cột 1: Ảnh gốc
        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 0].set_title(f"{domain_title}\n[Ảnh gốc]", fontsize=11)
        axes[row_idx, 0].axis("off")

        # Cột 2: Heatmap
        axes[row_idx, 1].imshow(heatmap_colored)
        axes[row_idx, 1].set_title("Grad-CAM Heatmap\n(Đỏ: Vùng kích hoạt cao)", fontsize=11)
        axes[row_idx, 1].axis("off")

        # Cột 3: Phủ Heatmap lên ảnh gốc
        axes[row_idx, 2].imshow(overlay_img)
        axes[row_idx, 2].set_title(f"Dự đoán: {pred_name}\nĐộ tin cậy: {conf:.1f}%", fontsize=11, fontweight="bold")
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu phân tích so sánh Grad-CAM tại: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Chạy phân tích Grad-CAM giải thích hiện tượng Shortcut Learning")
    parser.add_argument("--checkpoint", type=str, required=True, help="File trọng số .pth")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "mobilenet_v2"])
    parser.add_argument("--lab_image", type=str, required=True, help="Đường dẫn 1 ảnh phòng lab (PlantVillage)")
    parser.add_argument("--field_image", type=str, required=True, help="Đường dẫn 1 ảnh thực địa (PlantDoc)")
    parser.add_argument("--num_classes", type=int, default=17)
    parser.add_argument("--save_path", type=str, default="experiments/figures/gradcam_shortcut_comparison.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lấy danh sách tên class và cấu hình từ checkpoint
    chk = torch.load(args.checkpoint, map_location="cpu")
    class_names = chk.get("class_names", None)
    num_classes = len(class_names) if class_names else args.num_classes

    # Khởi tạo model tương ứng cấu hình và nạp weights
    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        use_cbam=chk.get("use_cbam", False),
        use_mixstyle=chk.get("use_mixstyle", False),
        use_concat_pool=chk.get("use_concat_pool", False),
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    # Lấy tầng conv cuối
    target_layer = model.get_last_conv_layer()
    gradcam = GradCAM(model=model, target_layer=target_layer)

    # Lấy danh sách tên class nếu có trong checkpoint
    chk = torch.load(args.checkpoint, map_location="cpu")
    class_names = chk.get("class_names", None)

    analyze_and_plot(
        model=model,
        gradcam=gradcam,
        img_path_lab=args.lab_image,
        img_path_field=args.field_image,
        class_names=class_names,
        device=device,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
