import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.datasets.plant_dataset import prepare_cross_domain_data, build_dataloaders
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import (
    compute_classification_metrics,
    plot_and_save_confusion_matrix,
    print_cross_domain_comparison,
)


@torch.no_grad()
def evaluate_dataloader(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels, _ in tqdm(dataloader, desc="Đang đánh giá", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_classification_metrics(all_labels, all_preds)
    return metrics, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Đánh giá chéo độc lập một checkpoint trên PlantDoc")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file .pth của mô hình")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "mobilenet_v2"])
    parser.add_argument("--source_dir", type=str, default="data/raw/plantvillage")
    parser.add_argument("--target_dir", type=str, default="data/raw/plantdoc/test")
    parser.add_argument("--mapping_file", type=str, default="data/class_mappings.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="experiments/figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Đang sử dụng thiết bị: {device}")

    # Chuẩn bị dữ liệu
    data_dict = prepare_cross_domain_data(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        mapping_file=args.mapping_file,
    )
    class_names = data_dict["canonical_classes"]
    num_classes = len(class_names)

    _, val_loader, target_loader = build_dataloaders(
        data_dict,
        image_size=(224, 224),
        batch_size=args.batch_size,
        num_workers=4,
        use_advanced_aug=False,
    )

    # Nạp mô hình
    model = build_model(model_name=args.model, num_classes=num_classes).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    # Đánh giá In-Domain
    print("\n--- ĐÁNH GIÁ TRÊN TẬP IN-DOMAIN (PLANTVILLAGE VAL) ---")
    in_metrics, y_val_true, y_val_pred = evaluate_dataloader(model, val_loader, device)

    # Đánh giá Cross-Domain
    print("\n--- ĐÁNH GIÁ TRÊN TẬP CROSS-DOMAIN (PLANTDOC REAL-WORLD) ---")
    out_metrics, y_tgt_true, y_tgt_pred = evaluate_dataloader(model, target_loader, device)

    # In kết quả so sánh
    ckpt_name = Path(args.checkpoint).stem
    print_cross_domain_comparison(in_metrics, out_metrics, model_name=ckpt_name)

    # Lưu biểu đồ
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    plot_and_save_confusion_matrix(
        y_tgt_true, y_tgt_pred, class_names,
        save_path=str(save_path / f"cm_eval_{ckpt_name}_target.png"),
        title=f"Cross-Domain Evaluation on PlantDoc ({ckpt_name})",
    )


if __name__ == "__main__":
    main()
