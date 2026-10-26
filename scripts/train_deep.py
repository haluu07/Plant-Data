import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

# Thêm root vào sys.path để import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.datasets.plant_dataset import prepare_cross_domain_data, build_dataloaders
from src.datasets.transforms import cutmix_data
from src.models import build_model
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logger import setup_logger
from src.utils.metrics import (
    compute_classification_metrics,
    plot_and_save_confusion_matrix,
    print_cross_domain_comparison,
)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, use_cutmix=False, cutmix_prob=0.3):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for images, labels, _ in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        is_cutmix = use_cutmix and (torch.rand(1).item() < cutmix_prob)

        if is_cutmix:
            images, targets_a, targets_b, lam = cutmix_data(images, labels)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(images)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

        # Backward & Optimize với Mixed Precision
        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total*100:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, desc="Eval"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, _ in tqdm(dataloader, desc=desc, leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    eval_loss = running_loss / total
    metrics = compute_classification_metrics(all_labels, all_preds)
    metrics["loss"] = eval_loss
    return metrics, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Deep Learning (ResNet-50 / MobileNetV2)")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "mobilenet_v2"], help="Tên kiến trúc mạng")
    parser.add_argument("--config", type=str, default=None, help="File cấu hình YAML (tùy chọn)")
    parser.add_argument("--source_dir", type=str, default="data/raw/plantvillage", help="Thư mục PlantVillage")
    parser.add_argument("--target_dir", type=str, default="data/raw/plantdoc/test", help="Thư mục PlantDoc test")
    parser.add_argument("--mapping_file", type=str, default="data/class_mappings.json", help="File ánh xạ lớp")
    parser.add_argument("--epochs", type=int, default=25, help="Số epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--use_advanced_aug", action="store_true", help="Bật Augmentation nâng cao (CutMix, Erasing, ColorJitter)")
    parser.add_argument("--num_workers", type=int, default=4, help="Số luồng nạp dữ liệu")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/checkpoints")
    args = parser.parse_args()

    # Tải cấu hình YAML nếu có
    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    elif Path(f"configs/{args.model}.yaml").exists():
        with open(f"configs/{args.model}.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg.get("training", {}).get("epochs", 25)
    batch_size = args.batch_size or cfg.get("data", {}).get("train_batch_size", 32)
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 0.001)

    exp_tag = f"{args.model}_adv_aug" if args.use_advanced_aug else f"{args.model}_standard"
    logger = setup_logger(exp_tag, log_file=f"experiments/logs/{exp_tag}.log")
    logger.info(f"=== BẮT ĐẦU HUẤN LUYỆN: {exp_tag.upper()} ===")

    # Thiết lập thiết bị tính toán
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[*] Sử dụng thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Chuẩn bị dữ liệu
    data_dict = prepare_cross_domain_data(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        mapping_file=args.mapping_file,
    )
    class_names = data_dict["canonical_classes"]
    num_classes = len(class_names)

    train_loader, val_loader, target_loader = build_dataloaders(
        data_dict,
        image_size=(224, 224),
        batch_size=batch_size,
        num_workers=args.num_workers,
        use_advanced_aug=args.use_advanced_aug,
    )

    # 2. Xây dựng mô hình
    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=True,
        dropout=cfg.get("model", {}).get("dropout", 0.2),
    ).to(device)

    # 3. Hàm mất mát & Tối ưu hóa
    label_smoothing = cfg.get("training", {}).get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer_name = cfg.get("training", {}).get("optimizer", "adamw").lower()
    if optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # 4. Vòng lặp huấn luyện
    best_val_acc = 0.0
    best_weights_path = Path(args.checkpoint_dir) / f"{exp_tag}_best.pth"

    logger.info(f"[*] Bắt đầu huấn luyện {epochs} epochs với lr={lr}, batch_size={batch_size}...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_cutmix=args.use_advanced_aug,
        )
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, desc="Val")
        scheduler.step()
        elapsed = time.time() - start_time

        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "best_acc": best_val_acc,
                    "class_names": class_names,
                    "use_advanced_aug": args.use_advanced_aug,
                },
                is_best=True,
                checkpoint_dir=args.checkpoint_dir,
                filename=f"{exp_tag}.pth",
            )

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)"
        )

    # 5. Đánh giá kiểm thử chéo trên mô hình tốt nhất
    logger.info("\n[*] Nạp trọng số tốt nhất để thực hiện Cross-Domain Evaluation...")
    load_checkpoint(str(best_weights_path), model, device=device)

    # In-Domain (PlantVillage Val)
    final_in_metrics, y_val_true, y_val_pred = evaluate(model, val_loader, criterion, device, desc="Final In-Domain")
    # Cross-Domain (PlantDoc Target Test)
    final_out_metrics, y_tgt_true, y_tgt_pred = evaluate(model, target_loader, criterion, device, desc="Cross-Domain")

    print_cross_domain_comparison(final_in_metrics, final_out_metrics, model_name=exp_tag.upper())

    # 6. Vẽ và lưu Ma trận nhầm lẫn
    fig_dir = Path("experiments/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_and_save_confusion_matrix(
        y_val_true, y_val_pred, class_names,
        save_path=str(fig_dir / f"cm_{exp_tag}_in_domain.png"),
        title=f"{exp_tag.upper()} - In-Domain (PlantVillage Val)",
    )
    plot_and_save_confusion_matrix(
        y_tgt_true, y_tgt_pred, class_names,
        save_path=str(fig_dir / f"cm_{exp_tag}_cross_domain.png"),
        title=f"{exp_tag.upper()} - Cross-Domain (PlantDoc Field)",
    )


if __name__ == "__main__":
    main()
