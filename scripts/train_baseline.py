import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import yaml

# Thêm root vào sys.path để import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.datasets.plant_dataset import prepare_cross_domain_data
from src.features.hog_extractor import FeatureExtractor
from src.models.svm_classifier import SVMModel
from src.utils.metrics import (
    compute_classification_metrics,
    plot_and_save_confusion_matrix,
    print_cross_domain_comparison,
)
from src.utils.logger import setup_logger


def extract_features_for_samples(samples, extractor, desc="Trích xuất đặc trưng"):
    X, y = [], []
    for path, label in tqdm(samples, desc=desc):
        try:
            feat = extractor.extract_from_path(path)
            X.append(feat)
            y.append(label)
        except Exception as e:
            continue
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Baseline HOG + SVM")
    parser.add_argument("--source_dir", type=str, default="data/raw/plantvillage", help="Thư mục PlantVillage")
    parser.add_argument("--target_dir", type=str, default="data/raw/plantdoc/test", help="Thư mục PlantDoc test")
    parser.add_argument("--mapping_file", type=str, default="data/class_mappings.json", help="File ánh xạ lớp")
    parser.add_argument("--config", type=str, default="configs/baseline_svm.yaml")
    parser.add_argument("--save_path", type=str, default="experiments/checkpoints/baseline_svm.joblib")
    parser.add_argument("--max_source_samples", type=int, default=200, help="Số mẫu tối đa mỗi class cho SVM (tránh tràn RAM/chạy chậm)")
    parser.add_argument("--max_target_samples", type=int, default=None, help="Số mẫu tối đa mỗi class cho target test")
    args = parser.parse_args()

    logger = setup_logger("Baseline_SVM", log_file="experiments/logs/baseline_svm.log")
    logger.info("=== BẮT ĐẦU HUẤN LUYỆN BASELINE: HOG + COLOR HISTOGRAM + SVM ===")

    # Đọc cấu hình
    cfg = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    # 1. Chuẩn bị dữ liệu và mapping lớp
    data_dict = prepare_cross_domain_data(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        mapping_file=args.mapping_file,
        max_source_per_class=args.max_source_samples,
        max_target_per_class=args.max_target_samples,
    )

    class_names = data_dict["canonical_classes"]

    # 2. Khởi tạo bộ trích xuất đặc trưng
    feat_cfg = cfg.get("feature_extraction", {})
    extractor = FeatureExtractor(
        image_size=tuple(feat_cfg.get("image_size", [128, 128])),
        hog_orientations=feat_cfg.get("hog", {}).get("orientations", 9),
        pixels_per_cell=tuple(feat_cfg.get("hog", {}).get("pixels_per_cell", [8, 8])),
        cells_per_block=tuple(feat_cfg.get("hog", {}).get("cells_per_block", [2, 2])),
        color_bins=feat_cfg.get("color_hist", {}).get("bins_per_channel", 16),
    )

    # 3. Trích xuất đặc trưng cho từng tập
    logger.info("[1/3] Trích xuất đặc trưng tập Train (PlantVillage)...")
    X_train, y_train = extract_features_for_samples(data_dict["train_samples"], extractor, desc="Train Features")

    logger.info("[2/3] Trích xuất đặc trưng tập Val (PlantVillage)...")
    X_val, y_val = extract_features_for_samples(data_dict["val_samples"], extractor, desc="Val Features")

    logger.info("[3/3] Trích xuất đặc trưng tập Target Test (PlantDoc)...")
    X_target, y_target = extract_features_for_samples(data_dict["target_test_samples"], extractor, desc="Target Test Features")

    # 4. Huấn luyện SVM
    model_cfg = cfg.get("model", {})
    svm_model = SVMModel(
        kernel=model_cfg.get("kernel", "rbf"),
        C=model_cfg.get("C", 10.0),
        gamma=model_cfg.get("gamma", "scale"),
    )
    svm_model.fit(X_train, y_train)
    svm_model.save(args.save_path)

    # 5. Đánh giá In-Domain (PlantVillage Val)
    y_pred_val = svm_model.predict(X_val)
    val_metrics = compute_classification_metrics(y_val, y_pred_val)
    logger.info(f"Kết quả In-Domain (PlantVillage): Accuracy = {val_metrics['accuracy']*100:.2f}%, F1 = {val_metrics['f1_macro']*100:.2f}%")

    # 6. Đánh giá Cross-Domain (PlantDoc Test)
    y_pred_target = svm_model.predict(X_target)
    target_metrics = compute_classification_metrics(y_target, y_pred_target)
    logger.info(f"Kết quả Cross-Domain (PlantDoc): Accuracy = {target_metrics['accuracy']*100:.2f}%, F1 = {target_metrics['f1_macro']*100:.2f}%")

    # 7. So sánh sụt giảm
    print_cross_domain_comparison(val_metrics, target_metrics, model_name="Baseline (HOG + SVM)")

    # 8. Lưu biểu đồ ma trận nhầm lẫn
    fig_dir = Path("experiments/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_and_save_confusion_matrix(
        y_val, y_pred_val, class_names,
        save_path=str(fig_dir / "cm_svm_in_domain.png"),
        title="Baseline SVM - In-Domain (PlantVillage Val)"
    )
    plot_and_save_confusion_matrix(
        y_target, y_pred_target, class_names,
        save_path=str(fig_dir / "cm_svm_cross_domain.png"),
        title="Baseline SVM - Cross-Domain (PlantDoc Test)"
    )


if __name__ == "__main__":
    main()
