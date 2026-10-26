import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def download_plantdoc(target_dir="data/raw/plantdoc"):
    """
    Tải tập dữ liệu thực địa PlantDoc từ GitHub chính thức.
    """
    target_path = Path(target_dir)
    if target_path.exists() and any(target_path.iterdir()):
        print(f"[+] Thư mục PlantDoc đã tồn tại tại: {target_dir}. Bỏ qua bước clone.")
        return

    print(f"[*] Đang tải tập dữ liệu thực địa PlantDoc từ GitHub chính thức...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", "https://github.com/pratikkayal/PlantDoc-Dataset.git", str(target_path)]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[+] Tải thành công PlantDoc vào: {target_dir}")
    else:
        print(f"[-] Lỗi khi tải PlantDoc. Vui lòng kiểm tra kết nối mạng.")


def download_plantvillage(target_dir="data/raw/plantvillage"):
    """
    Tải tập dữ liệu phòng lab PlantVillage trực tiếp từ GitHub của tác giả (Sharada Mohanty).
    Không cần tài khoản Kaggle hay token API.
    """
    target_path = Path(target_dir)
    if target_path.exists() and any(target_path.iterdir()):
        print(f"[+] Thư mục PlantVillage đã tồn tại tại: {target_dir}. Bỏ qua bước clone.")
        return

    temp_clone_dir = Path("data/raw/temp_plantvillage_repo")
    print(f"[*] Đang tải tập dữ liệu PlantVillage từ GitHub chính thức (Zero-auth)...")
    print("    Quá trình này mất khoảng 1-2 phút tùy tốc độ mạng của server...")

    if temp_clone_dir.exists():
        shutil.rmtree(temp_clone_dir)

    # Clone bản nông với --depth 1 để tải nhanh nhất
    cmd = [
        "git", "clone", "--depth", "1",
        "https://github.com/spMohanty/PlantVillage-Dataset.git",
        str(temp_clone_dir),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[-] Lỗi khi clone PlantVillage từ GitHub.")
        return

    # Di chuyển các thư mục màu gốc trong raw/color/ vào data/raw/plantvillage/
    source_color_dir = temp_clone_dir / "raw" / "color"
    if not source_color_dir.exists():
        # Fallback nếu cấu trúc nằm trực tiếp
        source_color_dir = temp_clone_dir / "color"

    target_path.mkdir(parents=True, exist_ok=True)
    for class_folder in source_color_dir.iterdir():
        if class_folder.is_dir():
            dest = target_path / class_folder.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(class_folder), str(dest))

    # Xóa thư mục tạm
    shutil.rmtree(temp_clone_dir, ignore_errors=True)
    print(f"[+] Tải và thiết lập thành công PlantVillage (38 lớp) vào: {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Tự động tải đầy đủ dữ liệu PlantDoc và PlantVillage")
    parser.add_argument("--plantdoc_dir", type=str, default="data/raw/plantdoc")
    parser.add_argument("--plantvillage_dir", type=str, default="data/raw/plantvillage")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("📥 BẮT ĐẦU TẢI DỮ LIỆU TỰ ĐỘNG CHO DỰ ÁN (ZERO AUTHENTICATION)")
    print("=" * 70)

    # 1. Tải PlantDoc
    download_plantdoc(args.plantdoc_dir)

    # 2. Tải PlantVillage
    download_plantvillage(args.plantvillage_dir)

    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT CHUẨN BỊ DỮ LIỆU!")
    print(f"   - PlantVillage: {args.plantvillage_dir}")
    print(f"   - PlantDoc:     {args.plantdoc_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
