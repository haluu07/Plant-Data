import argparse
import os
import subprocess
import sys
from pathlib import Path


def download_plantdoc(target_dir="data/raw/plantdoc"):
    """
    Tải tập dữ liệu PlantDoc chính thức từ repository GitHub của tác giả.
    """
    target_path = Path(target_dir)
    if target_path.exists() and any(target_path.iterdir()):
        print(f"[*] Thư mục PlantDoc đã tồn tại tại: {target_dir}. Bỏ qua bước clone.")
        return

    print(f"[*] Đang clone tập dữ liệu PlantDoc từ GitHub chính thức...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "https://github.com/pratikkayal/PlantDoc-Dataset.git", str(target_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[+] Tải thành công PlantDoc vào: {target_dir}")
    else:
        print(f"[-] Lỗi khi clone PlantDoc: {result.stderr}")


def print_instructions(plantvillage_dir="data/raw/plantvillage"):
    print("\n" + "=" * 70)
    print("           HƯỚNG DẪN TẢI TẬP DỮ LIỆU PLANTVILLAGE LÊN SERVER")
    print("=" * 70)
    print(f"1. Tải bằng Kaggle API (khuyên dùng):")
    print("   pip install kaggle")
    print("   kaggle datasets download -d abdallahalidev/plantvillage-dataset")
    print(f"   unzip plantvillage-dataset.zip -d {plantvillage_dir}")
    print("\n2. Hoặc clone từ GitHub chính thức:")
    print(f"   git clone https://github.com/spMohanty/PlantVillage-Dataset.git {plantvillage_dir}")
    print(f"   (Sau đó sử dụng thư mục: {plantvillage_dir}/raw/color/)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải và hướng dẫn chuẩn bị dữ liệu")
    parser.add_argument("--plantdoc_dir", type=str, default="data/raw/plantdoc")
    parser.add_argument("--plantvillage_dir", type=str, default="data/raw/plantvillage")
    args = parser.parse_args()

    download_plantdoc(args.plantdoc_dir)
    print_instructions(args.plantvillage_dir)
