"""Cấu hình mặc định cho các file .py ở local.

Data/review: chọn PART trong data_settings.py. Generate/train trên Kaggle: chỉnh
các cell cấu hình trong notebook, không cần sửa file này rồi upload lại.
config() lấy giá trị hiện tại; trong notebook chỉ gọi nó sau khi chọn đường dẫn/model.
RUN_NAME là folder kết quả thí nghiệm, không phải tên dataset hay tên kiến trúc.
"""

import os
import sys
from pathlib import Path

from data_settings import DATASET_NAME, PART, part_name

ROOT = Path(__file__).resolve().parent
CLEAN_DATASET = ROOT / "data_pipeline/exports" / DATASET_NAME / part_name()
GENERATION_NAME = part_name()
DATASET = ROOT / "datasets" / DATASET_NAME
# "all" = mọi part đã tải vào DATASET; hoặc ["vn-av-df-data-part1", ...].
DATASET_PARTS = [part_name()]
# Split train/validation/test đã gán ở 05_export (xem data_settings.py).
RUN_NAME = "train_part1_v002"  # Output: runs/<RUN_NAME>/<model>_seed<seed>/.
DEVICE = "cuda"  # Kaggle GPU. Use "cpu" for a local CPU-only environment.
RESUME = False
# generation/03b_merge.py: thư mục Output đã tải của từng phiên sinh (mỗi phiên một thư mục,
# chứa vn-av-df-forensics/), vd. [ROOT.parent / "kaggle_outputs" / n for n in ("gpu", "cpu", "gpu2")].
MERGE_INPUTS = []
MECHANISMS = ["mech_blend", "mech_lowres", "mech_vae", "mech_warp", "mech_temporal"]
# v002 §5.2: baseline có giám sát, đề xuất TriCue và biến thể bỏ từng nhánh.
ARCHITECTURES = [
    "avh_tcn",
    "p2_syncartifact",
    "tricue_mech",
    "tricue_w2l",
    "tricue_mech_w2l",
    "tricue_mech_no_lead",
    "tricue_mech_no_sync",
    "tricue_mech_no_art",
]
# v002 §5.3: chỉ nhánh dấu vết vẽ, phân tích cơ chế MechFake (rẻ).
# ARCHITECTURES = ["art_only_m1", ..., "art_drop_m5"]  # Xem cfg["methods"].
SEEDS = [42, 43, 44]  # Use [42] and EPOCHS=2 for a software smoke run.
EPOCHS = 20
HELD_OUT_GENERATORS = ["musetalk_1_5"]
AVH_PYTHON = os.environ.get(
    "VN_AVH_PYTHON",
    str(ROOT / ".venv-avhubert" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
)
MUSETALK_PYTHON = os.environ.get(
    "VN_MUSETALK_PYTHON",
    str(ROOT / ".venv-musetalk" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
)
CHECKPOINT = None  # None dùng DEMO_METHOD + seed đầu trong training.json, không xếp hạng model.
DEMO_METHOD = "tricue_mech"


def config():
    """Cấu hình dùng chung cho Run File và notebook; không hardcode thuật toán trong notebook."""
    cfg = {
        "seed": 42,
        "clean_dataset": str(CLEAN_DATASET),
        "generated_dataset": str(ROOT / "datasets" / DATASET_NAME / GENERATION_NAME),
        "data_part": PART,
        "dataset": str(DATASET),
        "dataset_parts": DATASET_PARTS,
        "plan": str(ROOT / "generation/plans" / f"{GENERATION_NAME}.json"),
        "root": str(ROOT),
        "merge_inputs": [str(p) for p in MERGE_INPUTS],
        "generation": {
            "clips_per_split": 2,
            # Độ dài đoạn partial lấy ngẫu nhiên liên tục trong khoảng (giây), mọi generator.
            "partial_range_s": [0.2, 3.0],
            "crf": 18,
            "max_side": 640,
            # Thiết kế 2×2: donor = tiếng thật khác của cùng người; source = vẽ theo tiếng gốc.
            # MechFake (mech_*) và chẩn đoán (diag_*) luôn giữ tiếng gốc.
            "fake_audio_modes_by_split": {
                "train": ["donor"],
                "validation": ["donor", "source"],
                "test": ["donor", "source"],
            },
            "include_sham": True,  # Sham = lệch A/V không AI, nhãn AI 0; cần cùng speaker_id.
            # Generator AI thật đứng đầu (real/sham đi kèm job đầu tiên của parent).
            "generators_by_split": {
                "train": ["wav2lip_gan", *MECHANISMS],
                "validation": ["wav2lip_gan", *MECHANISMS],
                "test": ["wav2lip_gan", "musetalk_1_5", "diag_paste", "diag_global"],
            },
        },
        "wav2lip": {
            "repo": str(ROOT / "external/Wav2Lip"),
            "checkpoint": str(ROOT / "weights/wav2lip/Wav2Lip-SD-GAN.pt"),
            "device": DEVICE,
            "python": sys.executable,
        },
        "mechfake": {
            "face_model": str(ROOT / "weights/media/yunet.onnx"),
            "vae": str(ROOT / "weights/sd-vae-ft-mse"),
            "device": DEVICE,
        },
        "device": DEVICE,
        "architectures": ARCHITECTURES,
        "seeds": SEEDS,
        "held_out_generators": HELD_OUT_GENERATORS,
        "cache": str(ROOT / "cache" / DATASET_NAME),
        "runs": str(ROOT / "runs" / RUN_NAME),
        "checkpoint": str(CHECKPOINT) if CHECKPOINT else None,
        "training": {
            "epochs": EPOCHS,
            "lr": 0.0003,
            "weight_decay": 0.0001,
            "hidden": 128,
            "dropout": 0.1,
            "patience": 5,
            "top_fraction": 0.1,
            "torch_threads": 2,
            "balance_parents": True,
            "artifact_aux_weight": 0.5,  # P2: head phụ artifact học cùng nhãn AI theo ô.
            "fusion_lr": 0.05,  # TriCue: head hợp điểm (vài tham số).
            "fusion_epochs": 60,
            "fusion_patience": 10,
        },
        "demo_output": str(ROOT / "outputs/demo"),
    }
    cfg["encoders"] = {
        "avhubert": {
            "kind": "avhubert",
            "repo": str(ROOT / "external/av_hubert"),
            "checkpoint": str(ROOT / "weights/avhubert/base_lrs3_iter4.pt"),
            "landmarks": str(ROOT / "weights/avhubert/shape_predictor_68_face_landmarks.dat"),
            "mean_face": str(ROOT / "weights/avhubert/20words_mean_face.npy"),
            "mean_face_url": "https://raw.githubusercontent.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/master/preprocessing/20words_mean_face.npy",
            "device": DEVICE,
            "python": AVH_PYTHON,
            "chunk_frames": 200,
            "overlap_frames": 50,
            "output_stride": 5,
            "max_duration": 60,
            "max_side": 640,
        },
        "dinov2": {
            "kind": "dinov2",
            "model": str(ROOT / "weights/dinov2-small"),  # facebook/dinov2-small, ViT-S/14.
            "device": DEVICE,
            "precision": "float16" if DEVICE == "cuda" else "float32",
            "crop_size": 224,  # Crop miệng RGB, bội số 14.
            "batch_size": 64,
            "max_duration": 60,
            "max_side": 640,  # Phải trùng AV-HuBERT để hộp miệng khớp frame.
        },
    }
    cfg["encoder"] = cfg["encoders"]["avhubert"]
    both = {"encoder": "avhubert", "artifact_encoder": "dinov2"}
    w2l, mech = ["wav2lip_gan"], ["mech_*"]
    cfg["methods"] = {
        "avh_tcn": {"encoder": "avhubert", "architecture": "tcn", "train_generators": w2l},
        "p2_syncartifact": {**both, "architecture": "syncartifact", "train_generators": w2l},
        "tricue_mech": {**both, "architecture": "tricue", "train_generators": mech},
        "tricue_w2l": {**both, "architecture": "tricue", "train_generators": w2l},
        "tricue_mech_w2l": {**both, "architecture": "tricue", "train_generators": mech + w2l},
        "tricue_mech_no_lead": {
            **both,
            "architecture": "tricue",
            "train_generators": mech,
            "branches": ["sync", "artifact"],
        },
        "tricue_mech_no_sync": {
            **both,
            "architecture": "tricue",
            "train_generators": mech,
            "branches": ["lead", "artifact"],
        },
        "tricue_mech_no_art": {
            **both,
            "architecture": "tricue",
            "train_generators": mech,
            "branches": ["sync", "lead"],
        },
    }
    # Phân tích cơ chế: nhánh dấu vết vẽ train trên một phần MechFake (M1…M5 = MECHANISMS).
    subsets = {"art_all": list(MECHANISMS)}  # Mốc: đủ 5 cơ chế.
    subsets.update({f"art_only_m{i + 1}": [name] for i, name in enumerate(MECHANISMS)})
    subsets["art_m1_m5"] = [MECHANISMS[0], MECHANISMS[4]]  # ≈ SAVe + AVPF đã công bố.
    subsets.update(
        {
            f"art_drop_m{i + 1}": [m for m in MECHANISMS if m != name]
            for i, name in enumerate(MECHANISMS)
        }
    )
    for name, generators in subsets.items():
        cfg["methods"][name] = {**both, "architecture": "artifact", "train_generators": generators}
    cfg["reconstruction"] = {"epochs": 20, "lr": 3e-4, "weight_decay": 1e-4, "patience": 5}
    # Stage S (P2 và TriCue): real (0), sham và real dịch lệch tiếng (1); không thấy fake.
    cfg["sync"] = {
        "epochs": 20,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "patience": 5,
        "shift_frames": [3, 15],
        "shift_probability": 0.5,
    }
    cfg["tricue"] = {"max_offset": 15, "lead_window": 12, "lead_lags": 8}
    cfg["lead"] = {"shrinkage": 0.1}
    cfg["musetalk"] = {
        "repo": str(ROOT / "external/MuseTalk"),
        "python": MUSETALK_PYTHON,
        "device": DEVICE,
        "batch_size": 4,
        "seed": 42,
    }
    cfg["demo_method"] = DEMO_METHOD
    cfg["bootstrap_replicates"] = 1000
    return cfg


def bootstrap():
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data_pipeline/src")]
