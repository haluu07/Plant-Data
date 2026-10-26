import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(y_true, y_pred):
    """
    Tính toán các chỉ số đánh giá phân loại toàn diện.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def plot_and_save_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """
    Vẽ ma trận nhầm lẫn chuẩn hóa và lưu thành file ảnh đẹp mắt.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Tỷ lệ chuẩn hóa"},
    )
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Nhãn Dự Đoán (Predicted)", fontsize=12)
    plt.ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ Confusion Matrix tại: {save_path}")


def print_cross_domain_comparison(in_metrics, out_metrics, model_name="Model"):
    """
    In bảng so sánh độ sụt giảm hiệu năng giữa In-domain và Cross-domain.
    """
    acc_in = in_metrics["accuracy"] * 100
    acc_out = out_metrics["accuracy"] * 100
    drop_acc = acc_in - acc_out

    f1_in = in_metrics["f1_macro"] * 100
    f1_out = out_metrics["f1_macro"] * 100
    drop_f1 = f1_in - f1_out

    print("\n" + "=" * 65)
    print(f"       KẾT QUẢ ĐÁNH GIÁ CHÉO (CROSS-DOMAIN BENCHMARK): {model_name}")
    print("=" * 65)
    print(f"| Chỉ số (Metric) | In-Domain (Lab) | Out-Domain (Field) | Suy giảm (Drop) |")
    print("-" * 65)
    print(f"| Accuracy        | {acc_in:6.2f}%        | {acc_out:6.2f}%            | -{drop_acc:5.2f}%        |")
    print(f"| F1-Score (Macro)| {f1_in:6.2f}%        | {f1_out:6.2f}%            | -{drop_f1:5.2f}%        |")
    print("=" * 65)
    if drop_acc > 40:
        print("⚠️ CẢNH BÁO: Hiện tượng Domain Shift & Shortcut Learning diễn ra rất mạnh!")
        print("   Mô hình đạt độ chính xác cao trong phòng lab nhờ đặc trưng nền/viền lá.")
    print("=" * 65 + "\n")
