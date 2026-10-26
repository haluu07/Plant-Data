import cv2
import numpy as np
from skimage.feature import hog


class FeatureExtractor:
    """
    Trích xuất đặc trưng truyền thống cho mô hình Baseline:
    1. HOG (Histogram of Oriented Gradients) - Bắt cấu trúc hình học, vân lá và rìa vết bệnh
    2. Color Histogram (HSV) - Bắt phân bố màu sắc đặc trưng của nấm và vi khuẩn
    """
    def __init__(
        self,
        image_size=(128, 128),
        hog_orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        color_bins=16,
    ):
        self.image_size = image_size
        self.hog_orientations = hog_orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.color_bins = color_bins

    def extract_from_path(self, image_path):
        """Đọc ảnh từ đường dẫn và trích xuất vector đặc trưng kết hợp"""
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        return self.extract(bgr)

    def extract(self, bgr_image):
        # Resize ảnh về kích thước cố định
        resized = cv2.resize(bgr_image, self.image_size)

        # 1. Trích xuất HOG trên ảnh xám
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hog_features = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True,
        )

        # 2. Trích xuất Color Histogram trên không gian màu HSV
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [self.color_bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [self.color_bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [self.color_bins], [0, 256])

        # Chuẩn hóa từng histogram
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        cv2.normalize(hist_v, hist_v)

        color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

        # Kết hợp HOG + Color Features
        combined = np.concatenate([hog_features, color_features])
        return combined
