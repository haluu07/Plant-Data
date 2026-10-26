import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMModel:
    """
    Mô hình Baseline: Chuẩn hóa đặc trưng (StandardScaler) + Bộ phân loại SVM.
    """
    def __init__(self, kernel="rbf", C=10.0, gamma="scale", random_state=42):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                probability=True,
                random_state=self.random_state,
            )),
        ])

    def fit(self, X, y):
        print(f"[*] Đang huấn luyện SVM (Kernel={self.kernel}, C={self.C}) trên {len(X)} mẫu...")
        self.pipeline.fit(X, y)
        print("[+] Huấn luyện SVM hoàn tất!")

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def save(self, save_path):
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, save_file)
        print(f"[+] Đã lưu mô hình SVM tại: {save_path}")

    @classmethod
    def load(cls, model_path):
        instance = cls()
        instance.pipeline = joblib.load(model_path)
        print(f"[+] Đã tải mô hình SVM từ: {model_path}")
        return instance
