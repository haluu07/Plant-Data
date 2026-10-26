import numpy as np
import torch
from torchvision import transforms


def get_transforms(
    image_size=(224, 224),
    is_train=True,
    use_advanced_aug=False,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Tạo pipeline tiền xử lý và tăng cường dữ liệu ảnh (Image Augmentation).
    
    Args:
        image_size: tuple (h, w)
        is_train: bool, True nếu là tập huấn luyện
        use_advanced_aug: bool, True nếu muốn kích hoạt Augmentation nâng cao chống bias nền
        mean: chuẩn hóa kênh màu
        std: chuẩn hóa độ lệch chuẩn
    """
    if not is_train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    if use_advanced_aug:
        # Augmentation nâng cao: Phá vỡ sự phụ thuộc vào màu nền và viền cắt nhân tạo
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), value="random"),
        ])
    else:
        # Augmentation tiêu chuẩn
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def cutmix_data(x, y, alpha=1.0):
    """
    Thực hiện kỹ thuật CutMix trên một batch tensor.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # Điều chỉnh lại lambda thực tế theo diện tích bbox
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y, y[index], lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Tâm ngẫu nhiên
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
