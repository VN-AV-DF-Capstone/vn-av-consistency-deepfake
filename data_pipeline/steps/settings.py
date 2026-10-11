"""Các đường dẫn dưới đây tự tính từ PART trong ../../data_settings.py.

Mỗi part một thư mục cho từng bước. Thêm URL vào videos.csv rồi chạy lại 01 → 05:
mỗi bước chỉ xử lý phần mới, giữ nguyên video đã tải, clip đã cắt và quyết định đã duyệt.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
# PART/SPLIT_* dùng ở 05_export (gán split khi export).
from data_settings import (  # noqa: E402, F401
    DATASET_NAME,
    MAX_CLIPS_PER_SPEAKER,
    MIN_TEST_GROUPS,
    PART,
    SPLIT_HISTORY,
    SPLIT_RATIOS,
    SPLIT_SEED,
    part_name,
)

DATASET_VERSION = part_name()
SOURCES = ROOT / "data/sources" / DATASET_NAME / DATASET_VERSION
RAW = ROOT / "data/raw" / DATASET_NAME / DATASET_VERSION
CUT = ROOT / "data/candidates" / DATASET_NAME / DATASET_VERSION
EXPORT = ROOT / "exports" / DATASET_NAME / DATASET_VERSION
