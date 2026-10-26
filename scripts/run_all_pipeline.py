import argparse
import subprocess
import sys


def run_command(cmd, desc):
    print("\n" + "=" * 70)
    print(f"[*] BƯỚC: {desc}")
    print(f"[*] LỆNH: {' '.join(cmd)}")
    print("=" * 70)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[-] Lỗi trong bước: {desc}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Tự động chạy toàn bộ quy trình thực nghiệm trên Server")
    parser.add_argument("--epochs", type=int, default=25, help="Số epochs cho Deep Learning")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_baseline", action="store_true", help="Bỏ qua huấn luyện SVM")
    args = parser.parse_args()

    # 1. Baseline HOG + SVM
    if not args.skip_baseline:
        run_command(
            [sys.executable, "scripts/train_baseline.py", "--max_source_samples", "300"],
            "1. Huấn luyện và Đánh giá chéo Baseline (HOG + SVM)",
        )

    # 2. Deep Learning: MobileNetV2 (Standard)
    run_command(
        [
            sys.executable, "scripts/train_deep.py",
            "--model", "mobilenet_v2",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
        ],
        "2. Huấn luyện MobileNetV2 Tiêu Chuẩn (In-domain & Cross-domain evaluation)",
    )

    # 3. Deep Learning: ResNet-50 (Standard)
    run_command(
        [
            sys.executable, "scripts/train_deep.py",
            "--model", "resnet50",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
        ],
        "3. Huấn luyện ResNet-50 Tiêu Chuẩn (In-domain & Cross-domain evaluation)",
    )

    # 4. Deep Learning: ResNet-50 + Advanced Augmentation (Chống Shortcut Bias)
    run_command(
        [
            sys.executable, "scripts/train_deep.py",
            "--model", "resnet50",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--use_advanced_aug",
        ],
        "4. Huấn luyện ResNet-50 với Advanced Augmentation (CutMix, Erasing, ColorJitter)",
    )

    # 5. Deep Learning: ResNet-50 + CBAM Attention (Triệt tiêu nhiễu nền)
    run_command(
        [
            sys.executable, "scripts/train_deep.py",
            "--model", "resnet50",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--use_cbam",
            "--use_concat_pool",
        ],
        "5. Huấn luyện ResNet-50 với CBAM Attention & Concat Pooling (Lọc sạch bối cảnh nền)",
    )

    # 6. Deep Learning: ResNet-50 + MixStyle + Adv Aug (SOTA Domain Generalization)
    run_command(
        [
            sys.executable, "scripts/train_deep.py",
            "--model", "resnet50",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--use_mixstyle",
            "--use_advanced_aug",
            "--use_cbam",
        ],
        "6. Huấn luyện ResNet-50 SOTA (MixStyle Domain Generalization + CBAM + Adv Aug)",
    )

    print("\n" + "=" * 70)
    print("🎉 TOÀN BỘ QUÁ TRÌNH THỰC NGHIỆM ĐÃ HOÀN TẤT THÀNH CÔNG!")
    print("   - Checkpoints đã lưu tại: experiments/checkpoints/")
    print("   - Biểu đồ Confusion Matrix & Kết quả tại: experiments/figures/")
    print("   - Log chi tiết tại: experiments/logs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
