"""Open and Run. Settings: ../settings.py (MERGE_INPUTS). Action: merge.

Gộp Output các phiên sinh độc lập (mỗi phiên một notebook Kaggle, đã tải về) vào
datasets/<dataset>/<part>/ + generation/plans/<part>.json, rồi chạy 04_review.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import bootstrap, config

bootstrap()
from vn_av_df.actions import execute

if __name__ == "__main__":
    execute("merge", config())
