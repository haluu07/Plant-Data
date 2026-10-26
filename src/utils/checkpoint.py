from pathlib import Path
import torch


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Lưu checkpoint của quá trình huấn luyện.
    """
    chk_dir = Path(checkpoint_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = chk_dir / filename
    torch.save(state, filepath)
    
    if is_best:
        best_path = chk_dir / filename.replace(".pth", "_best.pth")
        torch.save(state, best_path)
        print(f"[+] Đã lưu mô hình tốt nhất (Best Checkpoint): {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Tải weights từ checkpoint đã lưu.
    """
    print(f"[*] Đang tải checkpoint từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        
    epoch = checkpoint.get("epoch", 0)
    best_acc = checkpoint.get("best_acc", 0.0)
    print(f"[+] Tải thành công! (Epoch: {epoch}, Best Acc: {best_acc:.4f})")
    return epoch, best_acc
