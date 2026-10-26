#!/bin/bash
# ==============================================================================
# Script tự động thực thi toàn bộ quy trình Plant Disease & Domain Generalization
# ==============================================================================

set -e

echo "======================================================================"
echo "🚀 KHỞI ĐỘNG FULL PIPELINE TRÊN SERVER"
echo "======================================================================"

# 1. Cài đặt thư viện nếu chưa có
echo "[*] Kiểm tra và cài đặt thư viện..."
pip install -r requirements.txt

# 2. Chạy toàn bộ pipeline tự động
python3 scripts/run_all_pipeline.py --epochs 25 --batch_size 32

echo "======================================================================"
echo "✅ PIPELINE HOÀN TẤT THÀNH CÔNG!"
echo "======================================================================"
