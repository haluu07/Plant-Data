import json
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .transforms import get_transforms


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PlantDataset(Dataset):
    """
    Dataset loader cho ảnh bệnh cây trồng.
    Hỗ trợ nạp trực tiếp từ thư mục hoặc tự động lọc theo danh sách lớp chung.
    """
    def __init__(self, samples, transform=None):
        """
        Args:
            samples: list of tuples [(image_path, label_idx), ...]
            transform: torchvision transforms
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback nếu ảnh bị lỗi
            print(f"[Warning] Không thể đọc ảnh {path}: {e}")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(path)


def scan_dataset_samples(dataset_dir, target_classes=None, class_to_idx=None):
    """
    Quét toàn bộ ảnh trong dataset_dir (cấu trúc mỗi thư mục con là một class).
    
    Args:
        dataset_dir: Đường dẫn thư mục gốc chứa các thư mục con của từng lớp
        target_classes: Danh sách hoặc dict các lớp cần lấy (nếu None sẽ lấy tất cả)
        class_to_idx: Dict ánh xạ tên lớp sang index (0, 1, 2, ...)
        
    Returns:
        samples: list of (file_path, label_idx)
        class_to_idx: mapping dict
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {dataset_dir}")

    samples = []
    
    # Lấy danh sách các thư mục con
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    # Nếu chưa có class_to_idx, tự tạo
    if class_to_idx is None:
        if target_classes is not None:
            sorted_classes = sorted(list(target_classes))
        else:
            sorted_classes = sorted([d.name for d in subdirs])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted_classes)}

    for subdir in subdirs:
        class_name = subdir.name
        if class_name not in class_to_idx:
            continue
        
        idx = class_to_idx[class_name]
        for file in subdir.rglob("*"):
            if file.suffix.lower() in VALID_EXTENSIONS:
                samples.append((str(file), idx))

    return samples, class_to_idx


def prepare_cross_domain_data(
    source_dir,
    target_dir,
    mapping_file=None,
    val_split=0.2,
    seed=42,
    max_source_per_class=None,
    max_target_per_class=None,
):
    """
    Chuẩn bị dữ liệu cho bài toán Cross-Domain:
    - Source (PlantVillage): chia thành train và val
    - Target (PlantDoc): làm tập test ngoại miền (Out-Of-Domain test)
    
    Tự động căn chỉnh nhãn thông qua mapping_file (class_mappings.json).
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    source_class_to_idx = {}
    target_class_to_idx = {}
    canonical_classes = []

    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        shared_map = mapping_data.get("shared_classes", {})
        
        canonical_classes = sorted(list(shared_map.keys()))
        for idx, src_cls in enumerate(canonical_classes):
            tgt_cls = shared_map[src_cls]
            source_class_to_idx[src_cls] = idx
            target_class_to_idx[tgt_cls] = idx
    else:
        # Nếu không có mapping, lấy tất cả các class có tên giống nhau
        src_subdirs = {d.name for d in source_path.iterdir() if d.is_dir()}
        tgt_subdirs = {d.name for d in target_path.iterdir() if d.is_dir()}
        common = sorted(list(src_subdirs.intersection(tgt_subdirs)))
        canonical_classes = common
        for idx, cls_name in enumerate(common):
            source_class_to_idx[cls_name] = idx
            target_class_to_idx[cls_name] = idx

    print(f"[*] Số lượng lớp chung giữa 2 miền: {len(canonical_classes)}")

    # Quét dữ liệu Source (PlantVillage)
    source_samples, _ = scan_dataset_samples(
        source_dir, target_classes=source_class_to_idx.keys(), class_to_idx=source_class_to_idx
    )
    
    # Quét dữ liệu Target (PlantDoc)
    target_samples, _ = scan_dataset_samples(
        target_dir, target_classes=target_class_to_idx.keys(), class_to_idx=target_class_to_idx
    )

    # Giới hạn số lượng mẫu nếu cần (để test nhanh)
    if max_source_per_class is not None:
        source_samples = subsample_by_class(source_samples, max_source_per_class)
    if max_target_per_class is not None:
        target_samples = subsample_by_class(target_samples, max_target_per_class)

    # Chia Source thành Train và Val
    source_labels = [s[1] for s in source_samples]
    train_samples, val_samples = train_test_split(
        source_samples,
        test_size=val_split,
        random_state=seed,
        stratify=source_labels,
    )

    print(f"[*] Phân bố mẫu dữ liệu:")
    print(f"    - Source Train (PlantVillage): {len(train_samples)} ảnh")
    print(f"    - Source Val   (PlantVillage): {len(val_samples)} ảnh")
    print(f"    - Target Test  (PlantDoc)    : {len(target_samples)} ảnh (Kiểm thử thực địa)")

    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "target_test_samples": target_samples,
        "class_to_idx": source_class_to_idx,
        "canonical_classes": canonical_classes,
    }


def subsample_by_class(samples, max_per_class):
    """Giới hạn số mẫu tối đa mỗi class để cân bằng dữ liệu"""
    from collections import defaultdict
    class_buckets = defaultdict(list)
    for sample in samples:
        class_buckets[sample[1]].append(sample)
    
    balanced = []
    for cls_idx, items in class_buckets.items():
        balanced.extend(items[:max_per_class])
    return balanced


def build_dataloaders(
    data_dict,
    image_size=(224, 224),
    batch_size=32,
    num_workers=4,
    use_advanced_aug=False,
):
    """
    Xây dựng DataLoaders cho Train, Val, và Cross-Domain Target Test.
    """
    train_transform = get_transforms(image_size=image_size, is_train=True, use_advanced_aug=use_advanced_aug)
    eval_transform = get_transforms(image_size=image_size, is_train=False)

    train_ds = PlantDataset(data_dict["train_samples"], transform=train_transform)
    val_ds = PlantDataset(data_dict["val_samples"], transform=eval_transform)
    target_ds = PlantDataset(data_dict["target_test_samples"], transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    target_loader = DataLoader(
        target_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, target_loader
