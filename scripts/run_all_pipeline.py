import argparse
from pathlib import Path
import subprocess
import sys


def run_command(cmd, desc):
    print("\n" + "=" * 75)
    print(f"[*] BƯỚC: {desc}")
    print(f"[*] LỆNH: {' '.join(cmd)}")
    print("=" * 75)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[-] Lỗi trong bước: {desc}")
        sys.exit(res.returncode)


def auto_find_sample_images():
    """
    Tự động tìm 1 cặp ảnh cùng loại bệnh giữa PlantVillage và PlantDoc để phân tích Grad-CAM.
    """
    pv_dir = Path("data/raw/plantvillage")
    pd_dir = Path("data/raw/plantdoc/test")

    # Thử tìm Tomato Early Blight
    pv_candidates = list(pv_dir.glob("Tomato___Early_blight/*.*")) or list(pv_dir.glob("*Tomato*/*.*")) or list(pv_dir.glob("*/*.*"))
    pd_candidates = list(pd_dir.glob("*Tomato*Early*blight*/*.*")) or list(pd_dir.glob("*/*.*"))

    if pv_candidates and pd_candidates:
        return str(pv_candidates[0]), str(pd_candidates[0])
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Chạy FULL PIPELINE nghiên cứu từ A đến Z bằng 1 câu lệnh duy nhất")
    parser.add_argument("--epochs", type=int, default=25, help="Số epochs cho mỗi mô hình Deep Learning")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--skip_baseline", action="store_true", help="Bỏ qua huấn luyện SVM")
    parser.add_argument("--skip_download", action="store_true", help="Bỏ qua bước kiểm tra dữ liệu")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("🚀 BẮT ĐẦU CHẠY TOÀN BỘ QUY TRÌNH (FULL PIPELINE EXECUTION)")
    print("   Plant Disease Recognition and Cross-Domain Generalization")
    print("=" * 75)

    # BƯỚC 0: TỰ ĐỘNG KIỂM TRA & TẢI DỮ LIỆU
    if not args.skip_download:
        plantdoc_path = Path("data/raw/plantdoc")
        if not plantdoc_path.exists() or not any(plantdoc_path.iterdir()):
            run_command([sys.executable, "scripts/download_data.py"], "0. Tự động tải tập dữ liệu thực địa PlantDoc")

        plantvillage_path = Path("data/raw/plantvillage")
        if not plantvillage_path.exists() or not any(plantvillage_path.iterdir()):
            print("⚠️ CẢNH BÁO: Chưa tìm thấy dữ liệu PlantVillage trong 'data/raw/plantvillage/'")
            print("   Vui lòng giải nén tập PlantVillage vào thư mục 'data/raw/plantvillage/' rồi chạy lại.")
            sys.exit(1)

    # BƯỚC 1: BASELINE HOG + SVM
    if not args.skip_baseline:
        run_command(
            [sys.executable, "scripts/train_baseline.py", "--max_source_samples", "300"],
            "1/6. Huấn luyện & Đánh giá chéo Baseline (HOG + Color Histogram + SVM)",
        )

    # BƯỚC 2: DEEP LEARNING - MOBILENETV2 (STANDARD)
    run_command(
        [
            sys.executable, "scripts/train_deep.py",
            "--model", "mobilenet_v2",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size * 2),
        ],
        "2/6. Huấn luyện MobileNetV2 Tiêu Chuẩn (Edge/IoT Architecture)",
    )

    # BƯỚC 3: DEEP LEARNING - RESNET-50 (STANDARD)
    run_command(
        [
            sys.executable, "scripts/train_deep.py",
            "--model", "resnet50",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
        ],
        "3/6. Huấn luyện ResNet-50 Tiêu Chuẩn (Deep Residual Backbone)",
    )

    # BƯỚC 4: DEEP LEARNING - RESNET-50 + ADVANCED AUGMENTATION
    run_command(
        [
            sys.executable, "scripts/train_deep.py",
            "--model", "resnet50",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--use_advanced_aug",
        ],
        "4/6. Huấn luyện ResNet-50 với Advanced Augmentation (CutMix + Erasing + ColorJitter)",
    )

    # BƯỚC 5: DEEP LEARNING - RESNET-50 + CBAM ATTENTION
    run_command(
        [
            sys.executable, "scripts/train_deep.py",
            "--model", "resnet50",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--use_cbam",
            "--use_concat_pool",
        ],
        "5/6. Huấn luyện ResNet-50 với CBAM Attention & Concat Pooling (Triệt tiêu nhiễu nền)",
    )

    # BƯỚC 6: DEEP LEARNING - RESNET-50 SOTA (MIXSTYLE + CBAM + AUG)
    run_command(
        [
            sys.executable, "scripts/train_deep.py",
            "--model", "resnet50",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--use_mixstyle",
            "--use_advanced_aug",
            "--use_cbam",
            "--use_concat_pool",
        ],
        "6/6. Huấn luyện ResNet-50 SOTA Toàn Diện (MixStyle Domain Generalization + CBAM + CutMix)",
    )

    # BƯỚC 7: TỰ ĐỘNG CHẠY GRAD-CAM XUẤT ẢNH BÁO CÁO
    lab_img, field_img = auto_find_sample_images()
    best_sota_ckpt = "experiments/checkpoints/resnet50_adv_aug_cbam_mixstyle_concatpool_best.pth"
    if not Path(best_sota_ckpt).exists():
        best_sota_ckpt = "experiments/checkpoints/resnet50_standard_best.pth"

    if lab_img and field_img and Path(best_sota_ckpt).exists():
        run_command(
            [
                sys.executable, "scripts/run_gradcam_analysis.py",
                "--checkpoint", best_sota_ckpt,
                "--model", "resnet50",
                "--lab_image", lab_img,
                "--field_image", field_img,
                "--save_path", "experiments/figures/gradcam_shortcut_comparison.png",
            ],
            "7/7. Trực quan hóa Grad-CAM Heatmap so sánh hiện tượng Lab vs Field",
        )

    print("\n" + "=" * 75)
    print("🎉🎉🎉 TOÀN BỘ QUY TRÌNH THỰC NGHIỆM ĐÃ HOÀN TẤT THÀNH CÔNG! 🎉🎉🎉")
    print("=" * 75)
    print("📁 Danh mục kết quả được tạo ra:")
    print("   1. Checkpoints mô hình đã lưu tại:  experiments/checkpoints/")
    print("   2. Ma trận nhầm lẫn & ảnh Grad-CAM: experiments/figures/")
    print("   3. Toàn bộ nhật ký chạy tại:        experiments/logs/")
    print("\n👉 Để khởi chạy Web App Demo xem kết quả trực quan:")
    print("   python app/app.py --port 7860 --share")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
