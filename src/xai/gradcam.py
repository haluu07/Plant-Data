import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping).
    Trực quan hóa vùng mà mạng nơ-ron tập trung vào khi đưa ra dự đoán.
    Giúp phát hiện hiện tượng học đường tắt (Shortcut Learning) và background bias.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Đăng ký hooks để lấy activation và gradient
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Sinh ra Grad-CAM heatmap cho ảnh đầu vào.
        
        Args:
            input_tensor: torch.Tensor [1, 3, H, W]
            target_class: index của lớp cần giải thích (nếu None sẽ lấy lớp có logit cao nhất)
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Tính toán trọng số alpha_k = Global Average Pooling của gradients
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # [C, 1, 1]

        # Tổ hợp tuyến tính của activation maps có trọng số
        cam = torch.sum(weights * activations, dim=0)  # [H, W]

        # Áp dụng hàm ReLU để chỉ giữ lại các đặc trưng có đóng góp tích cực
        cam = F.relu(cam)

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        # Chuẩn hóa về [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class


def overlay_cam_on_image(img_pil, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Phủ bản đồ nhiệt (Heatmap) lên ảnh gốc để hiển thị trực quan.
    """
    img_np = np.array(img_pil)
    h, w, _ = img_np.shape

    # Resize CAM theo kích thước ảnh gốc
    heatmap = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = np.float32(heatmap_colored) * alpha + np.float32(img_np) * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay), Image.fromarray(heatmap_colored)
