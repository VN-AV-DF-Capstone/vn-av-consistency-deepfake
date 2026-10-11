"""Tự tải source/checkpoint và tài sản phụ trợ, không train và không cài worker.

Run File: tải encoder của ARCHITECTURES trong settings.py.
Chỉ AV-HuBERT: python training/00_setup.py --encoder avhubert
Chỉ DINOv2:    python training/00_setup.py --encoder dinov2  (nhánh dấu vết vẽ, cần thêm avhubert)
Cần Internet và Git. Không cần dataset hoặc GPU ở bước tải này.
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import RESUME, bootstrap, config

bootstrap()
from vn_av_df.actions import execute

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download encoder source, weights and preprocessing assets."
    )
    parser.add_argument(
        "--encoder",
        choices=["selected", "avhubert", "dinov2"],
        default="selected",
        help="selected: encoders needed by ARCHITECTURES in settings.py",
    )
    args = parser.parse_args()
    cfg = config()
    if args.encoder != "selected":
        # Chỉ đổi phạm vi tải của lệnh này, không sửa lựa chọn train trong settings.py.
        cfg.pop("methods", None)
        cfg["encoder"] = cfg["encoders"][args.encoder]
    execute("encoder_setup", cfg, RESUME)
