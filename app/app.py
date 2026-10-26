import argparse
import sys
from pathlib import Path
import gradio as gr
import numpy as np
from PIL import Image
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.datasets.transforms import get_transforms
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.xai.gradcam import GradCAM, overlay_cam_on_image


# Biến toàn cục lưu trữ model
CURRENT_MODEL = None
CURRENT_GRADCAM = None
CLASS_NAMES = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_instance(model_name, checkpoint_path):
    global CURRENT_MODEL, CURRENT_GRADCAM, CLASS_NAMES

    if not Path(checkpoint_path).exists():
        return f"Không tìm thấy file checkpoint: {checkpoint_path}"

    chk = torch.load(checkpoint_path, map_location="cpu")
    CLASS_NAMES = chk.get("class_names", [f"Class {i}" for i in range(17)])
    num_classes = len(CLASS_NAMES)

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        use_cbam=chk.get("use_cbam", False),
        use_mixstyle=chk.get("use_mixstyle", False),
        use_concat_pool=chk.get("use_concat_pool", False),
    ).to(DEVICE)
    load_checkpoint(checkpoint_path, model, device=DEVICE)
    model.eval()

    target_layer = model.get_last_conv_layer()
    gradcam = GradCAM(model=model, target_layer=target_layer)

    CURRENT_MODEL = model
    CURRENT_GRADCAM = gradcam
    return f"Đã nạp thành công mô hình {model_name.upper()} ({num_classes} lớp)"


def predict(image, model_choice, checkpoint_file):
    global CURRENT_MODEL, CURRENT_GRADCAM, CLASS_NAMES

    if image is None:
        return None, None, "Vui lòng tải lên một ảnh lá cây!"

    if CURRENT_MODEL is None or checkpoint_file:
        status = load_model_instance(model_choice, checkpoint_file)
        if "Không tìm thấy" in status:
            return None, None, status

    raw_img = Image.fromarray(image).convert("RGB")
    transform = get_transforms(image_size=(224, 224), is_train=False)
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = CURRENT_MODEL(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    # Lấy top-3 dự đoán
    top3_indices = np.argsort(probs)[::-1][:3]
    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in top3_indices}

    # Sinh Grad-CAM cho class cao nhất
    top1_idx = top3_indices[0]
    cam, _ = CURRENT_GRADCAM.generate(input_tensor, target_class=top1_idx)
    overlay_img, heatmap_colored = overlay_cam_on_image(raw_img, cam, alpha=0.55)

    info_text = (
        f"**Kết quả phân loại chính**: `{CLASS_NAMES[top1_idx]}` ({probs[top1_idx]*100:.2f}%)\n\n"
        f"**Đánh giá Grad-CAM**: Vùng màu đỏ/cam thể hiện khu vực kích hoạt mạnh nhất của mạng nơ-ron. "
        f"Nếu vùng kích hoạt nằm ngoài rìa lá hoặc trên phông nền, mô hình đang gặp hiện tượng **Background Bias (Học đường tắt)**."
    )

    return overlay_img, confidences, info_text


def create_ui():
    with gr.Blocks(title="Plant Disease Recognition & Cross-Domain XAI") as demo:
        gr.Markdown(
            """
            # 🌿 Plant Disease Recognition & Generalization Demo
            ### Nhận dạng bệnh cây trồng & Trực quan hóa vùng kích hoạt mạng với Grad-CAM
            Hệ thống hỗ trợ kiểm tra xem mạng nơ-ron nhận diện bệnh dựa trên **vết tổn thương thật sự** hay bị **nhiễu nền (Background Bias)**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="numpy", label="Ảnh lá cây (Lab hoặc Thực địa ngoài vườn)")
                model_choice = gr.Dropdown(
                    choices=["resnet50", "mobilenet_v2"],
                    value="resnet50",
                    label="Kiến trúc mô hình",
                )
                ckpt_input = gr.Textbox(
                    value="experiments/checkpoints/resnet50_standard_best.pth",
                    label="Đường dẫn Checkpoint (.pth)",
                )
                btn_run = gr.Button("Phân Tích & Dự Đoán", variant="primary")

            with gr.Column(scale=1):
                cam_output = gr.Image(type="pil", label="Grad-CAM Visual Explanation (Heatmap Overlay)")
                label_output = gr.Label(num_top_classes=3, label="Top 3 Dự Đoán")
                info_output = gr.Markdown()

        btn_run.click(
            fn=predict,
            inputs=[image_input, model_choice, ckpt_input],
            outputs=[cam_output, label_output, info_output],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Tạo public link qua Gradio share")
    args = parser.parse_args()

    ui = create_ui()
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
