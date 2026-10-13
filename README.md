# VN-AV-DF Forensics

Phát hiện và định vị lip-sync deepfake trong video tiếng Việt: **video YouTube → cắt/duyệt → chia split → sinh fake → trích feature → train + test → so sánh → demo**.
Điểm 0-1 là mức nghi ngờ, chưa hiệu chuẩn thành xác suất; đoạn không quan sát được bị mask, không coi là real.

## Tổng quan

| # | Bước | Chạy ở đâu | Entrypoint | Đầu ra (đường dẫn trong repo) |
|---|---|---|---|---|
| 1 | Thu thập → tải → cắt → duyệt → export (gán split) | Local (cắt có thể Kaggle) | `data_pipeline/steps/01…05`, [cut.ipynb](notebooks/cut.ipynb) | `data_pipeline/exports/vn-av-df-data/<part>/` |
| 2 | Sinh real/fake/sham | Kaggle | [generate.ipynb](notebooks/generate.ipynb) | `datasets/vn-av-df-data/<part>/` + `generation/plans/<part>.json` |
| 3 | Duyệt mẫu sinh → chốt → ZIP | Local | `generation/04…06` | `datasets/vn-av-df-data/<part>.zip` |
| 4 | Trích feature (cache) | Kaggle | [prepare.ipynb](notebooks/prepare.ipynb) | `cache/vn-av-df-data/<encoder>/` |
| 5 | Train + test | Kaggle (hoặc local) | [train.ipynb](notebooks/train.ipynb), `training/02…03` | `runs/<RUN_NAME>/` |
| 6 | So sánh kiến trúc theo part | Local hoặc Kaggle | [compare.ipynb](notebooks/compare.ipynb) | `reports/compare/` |
| 7 | Demo | Local | `training/04_demo.py` | http://127.0.0.1:8000 |

**Part** (`vn-av-df-data-part1`, `-part2`, …) là một đợt dữ liệu bổ sung, xử lý và đóng gói riêng. Local chọn part bằng `PART` trong [data_settings.py](data_settings.py); Kaggle chỉnh trong cell cấu hình của notebook.

### Quy ước tên và đường dẫn (Local ↔ Kaggle)

Mọi notebook đọc input bằng **một đường dẫn đầy đủ** (copy từ panel *Input* của Kaggle; nếu ZIP giữ thêm một thư mục bên trong, notebook tự tìm vào) và ghi Output **dưới `vn-av-df-forensics/` đúng đường dẫn như ở local**, chỉ giữ thư mục kết quả. Tải Output về là đặt thẳng vào repo.

| Dữ liệu | Thư mục local | Kaggle dataset = tên ZIP | Tạo ZIP |
|---|---|---|---|
| Raw part | `data_pipeline/data/raw/vn-av-df-data/vn-av-df-data-partN/` | `vn-av-df-raw-partN` | nén thư mục part (có `videos/`, `sources.jsonl`, `logs/`) |
| Part sạch (có split) | `data_pipeline/exports/vn-av-df-data/vn-av-df-data-partN/` | `vn-av-df-clean-partN` | nén thư mục part |
| Part đã sinh, đã chốt | `datasets/vn-av-df-data/vn-av-df-data-partN/` | `vn-av-df-data-partN` | `generation/06_export.py` (giữ thư mục `vn-av-df-data-partN/` bên trong) |
| Cache feature | `cache/vn-av-df-data/{avhubert,dinov2}/` | `vn-av-df-features-partN` | nén các thư mục encoder (lệnh ở mục 4) |

- **Tên thư mục part (`vn-av-df-data-partN`) đi vào hash lựa chọn dữ liệu.** Luôn upload part đã sinh bằng ZIP của `06_export.py`; đổi tên thư mục part làm cache/checkpoint không khớp.
- `PREVIOUS_OUTPUT` (cut, generate) trỏ tới thư mục `vn-av-df-forensics` trong Output phiên trước để chạy tiếp.
- Log W&B nằm ở `/kaggle/working/wandb` (vẫn trong Output, `wandb sync` lại được).

### Giới hạn Kaggle cần nhớ

| Giới hạn | Ảnh hưởng | Cách làm |
|---|---|---|
| 12 giờ/phiên, ~30 giờ GPU/tuần, Output ≤ 20 GB | Sinh (Wav2Lip, MuseTalk, MechFake) và trích AV-HuBERT là phần tốn nhất | Chia phiên theo generator, MechFake không VAE chạy phiên **CPU** (không trừ quota GPU, mục 2); generate có `TIME_BUDGET_HOURS`, chạy tiếp bằng `PREVIOUS_OUTPUT`; prepare trích tiếp bằng `CACHE_INPUTS` |
| Dataset tạo từ Output chỉ lấy **500 file** | Cache ~6000 file/encoder, part đã sinh hàng nghìn clip | Tải Output về, ZIP ở local rồi upload thành dataset |
| `kaggle kernels output` chỉ lấy version mới nhất, skip chậm, đứt mạng là dừng | Tải Output lớn hay lỗi | `python tools/kaggle_pull.py <owner/slug> .. "<regex>"` (bỏ qua file đã có, tự thử lại) |
| Kaggle CLI mới cần Python ≥ 3.11 | Env dự án v002 đã là 3.11 | `python -m pip install -U kaggle`, `kaggle auth login` |
| `mech_vae` giữ bộ đệm GPU, worker MuseTalk mở sau trên cùng GPU báo `CUDA out of memory` (phiên 3 part1) | Đã sửa: lô VAE co theo cạnh ảnh (16 ảnh ở 256 px, 2 ở 600 px) + trả bộ đệm sau mỗi clip (`mechfake.py`, không đổi chữ ký output) | Phiên đã lỗi: chạy tiếp `GENERATORS = ["musetalk_1_5"]` với `PREVIOUS_OUTPUT` = Output phiên đó |

## Cài đặt local

Python **3.11** (env conda `vn-av-df`, v002 nâng từ 3.10), chạy từ thư mục `vn-av-df-forensics`. Worker AV-HuBERT/MuseTalk là env riêng (Python 3.10, [environments/README.md](environments/README.md)), không đổi. Notebook Kaggle tự cài thư viện và tải model.

```powershell
python -m pip install -e ./data_pipeline -e ".[training]"   # data, duyệt, train/test CPU, demo
python -m pip install pytest httpx                           # (tuỳ chọn) chạy kiểm thử; hoặc dùng extra [dev]
python data_pipeline/steps/00_setup.py                       # YuNet + Silero VAD
python training/00_setup.py --encoder avhubert               # demo B-AVH/P2/TriCue (cần worker, mục 7)
python training/00_setup.py --encoder dinov2                 # demo P2/TriCue
```

`python 00_install.py` (`PROFILE = "data"` hoặc `"demo"`) thay được lệnh pip đầu. Không cần Wav2Lip/MuseTalk ở local.

## 1. Dữ liệu real và split

Điền `data_pipeline/data/sources/vn-av-df-data/<part>/videos.csv` (xem [mẫu](data_pipeline/configs/videos.example.csv)):

```csv
url,speaker_id
https://www.youtube.com/playlist?list=PLxxxxxxxx,speaker_01
https://www.youtube.com/watch?v=xxxxxxxxxxx,speaker_02
```

`url` là video hoặc playlist (mọi video trong playlist nhận `speaker_id` của dòng). Giữ `speaker_id` nhất quán giữa các part. Chạy lần lượt trong `data_pipeline/steps/`:

| Bước | Việc làm | Kết quả chính |
|---|---|---|
| `01_collect.py` | Bung playlist, bỏ trùng, hỏi metadata YouTube, kiểm chất lượng | `selected_videos.csv`, `video_metadata.csv`, `skipped_videos.csv` |
| `02_download.py` | Tải bản ≤1080 (cạnh ngắn), kiểm lại file | `data/raw/<dataset>/<part>/` |
| `03_cut.py` | VAD (gộp lặng < 0,6 s) + dò mặt 4 fps rồi **kiểm mặt mọi frame** của cửa sổ giữ lại, cắt clip 5-8 s, 25 fps CFR, cạnh ngắn ≤1080 | `data/candidates/<dataset>/<part>/` |
| `04_review.py` | Duyệt tại http://127.0.0.1:8001 | `review.csv` |
| `05_export.py` | Dựng part sạch từ clip keep **và gán split** train/validation/test | `exports/<dataset>/<part>/` (+ `split-lock.json`) |

- **Split** gán ở `05_export` theo nhóm liên thông người/nguồn/hash (cả nhóm cùng một split), tỷ lệ `SPLIT_RATIOS` (v002: **60/15/25**), `SPLIT_SEED` trong `data_settings.py`; cần ≥3 nhóm độc lập. Test ít hơn `MIN_TEST_GROUPS` (5) nhóm thì export **cảnh báo** (`quality_report.json` → `warnings`). `MAX_CLIPS_PER_SPEAKER` (mặc định tắt) giữ tối đa N clip mỗi speaker. Mỗi part chia độc lập; muốn part mới kế thừa split part cũ thì điền `SPLIT_HISTORY` bằng `split-lock.json` của các part trước. Đổi split = export lại và sinh lại part.
- **Trang duyệt** (04): danh sách có lọc theo quyết định/speaker/nguồn, phím tắt `K` giữ, `R` loại, `U` chưa rõ, `←/→`, `N` tới clip chưa chắc, `Space` phát/dừng, `L` lặp; chỉnh tốc độ phát. Clip mới cắt ở trạng thái *chưa rõ*.
- **Kiểm chất lượng (01):** loại video private/đã xoá/livestream, fps gốc <25, cạnh ngắn <720 px, dài <5 s hoặc >12 giờ. `skipped_videos.csv` ghi lý do; `speaker_conflict = yes` → sửa `videos.csv` trước khi tải.
- **Mạng:** lỗi IPv6 thì thêm `--force-ipv4`; bị chặn 403 thì thêm `--cookies-from-browser firefox`.
- **Bổ sung video:** thêm URL rồi chạy lại 01 → 05; mỗi bước chỉ xử lý phần mới, giữ quyết định đã duyệt. **Đổi luật cắt** (`configs/data.yaml`, code, model) thì 03 báo lỗi: xoá `data/candidates/<dataset>/<part>` để cắt lại.

**Cắt trên Kaggle** (khi máy chậm): upload thư mục raw thành `vn-av-df-raw-partN` → chạy [cut.ipynb](notebooks/cut.ipynb) (`RAW_PART`) → tải `vn-av-df-forensics/data_pipeline/data/candidates/...` về đúng chỗ → `04_review.py`. Code/config cắt phải giống local (có chữ ký); hết phiên thì đặt `PREVIOUS_OUTPUT` để cắt tiếp.

## 2. Sinh fake (Kaggle)

Thiết kế **2×2**:

| | Không dấu vết AI | Có dấu vết AI |
|---|---|---|
| **Tiếng khớp miệng** | real | fake `source`: miệng vẽ theo tiếng gốc (chỉ có artifact) |
| **Tiếng lệch miệng** | sham: ghép tiếng thật khác của cùng người, nhãn AI = 0 | fake `donor`: miệng vẽ theo tiếng thật khác của cùng người (mối đe doạ chính) |

1. **Push code lên GitHub** (notebook clone `REPO_URL`), upload part sạch thành `vn-av-df-clean-partN`; trong [generate.ipynb](notebooks/generate.ipynb) đặt `PART`, `CLEAN_PART`.
2. **Chạy thử trước:** `GENERATORS = "all"`, `CLIPS_PER_SPLIT = 2` (2 parent mỗi split, ~20 phút) để bắt lỗi trước khi chạy thật. Output chạy thử **không** dùng làm `PREVIOUS_OUTPUT` cho bản thật (plan khác).
3. **Chạy thật** (`CLIPS_PER_SPLIT = 0`): **3 phiên độc lập**, mỗi phiên một notebook (notebook `generate` và các bản *Copy & Edit*, vd. `generate-cpu`), đều `PREVIOUS_OUTPUT = None`; tải 3 Output về rồi **gộp ở local** (`03b_merge.py`, mục 3). Mọi phiên giữ nguyên mọi thông số khác (plan khoá cấu hình); chỉ đổi `GENERATORS` và Accelerator:

   | Phiên | Accelerator | `GENERATORS` | Part1 (ước tính) |
   |---|---|---|---|
   | 1 | GPU T4 ×2 | `["wav2lip_gan"]` | 804 cặp, ~8 h |
   | 2 | **None** (CPU, không trừ quota GPU) | `["mech_blend", "mech_lowres", "mech_warp", "mech_temporal", "diag_paste", "diag_global"]` | 1740 + 280 cặp |
   | 3 | GPU T4 ×2 | `["musetalk_1_5", "mech_vae"]` | 280 + 435 cặp |

   - Mỗi phiên tự lập plan (cùng seed + part sạch + cấu hình → **trùng nhau**) và chỉ sinh cặp của generator phiên đó; thứ tự và chạy song song tuỳ ý. Output mỗi phiên chỉ chứa phần của nó, không trùng dữ liệu trên Kaggle.
   - **Không đổi code từ phiên đầu đến phiên cuối**: `generation.json` khoá hash `generation.py`, `render.py` (và `mechfake.py` theo generator); lệch → không gộp được, phải sinh lại phiên đó.
   - Hết `TIME_BUDGET_HOURS` (11 h) → `status: partial`: chạy lại **đúng phiên đó trong notebook của nó** với `PREVIOUS_OUTPUT` = Output vừa lưu (*Add Input → Your Work*, trỏ tới thư mục `vn-av-df-forensics`). Phiên xong khi `remaining_by_generator` không còn generator của nó (vẫn `partial` vì cặp của phiên khác).
   - `PREVIOUS_OUTPUT` dạng danh sách = gộp ngay trên Kaggle (vd. phiên 3 gắn Output phiên 1 + 2 rồi sinh nốt): một Output đủ cả part, nhưng Kaggle lưu trùng dữ liệu.
   - Real/sham sinh kèm Wav2Lip (phiên 1); MechFake của parent mà cặp Wav2Lip bị bỏ (không thấy mặt) bị loại khi gộp.
   - `DEVICE = "auto"`: không GPU thì chạy CPU với `CPU_WORKERS` (2) cặp song song; phiên CPU chỉ nhận `mech_*` trừ `mech_vae` và `diag_*`, generator khác báo lỗi ngay.
   - Muốn gọn (một notebook, tuần tự): `GENERATORS = "all"` qua ~2 phiên GPU (~15–17 h), mỗi phiên sau `PREVIOUS_OUTPUT` = Output phiên trước.
   - Chạy nền: *Save Version → Save & Run All*, Output tự lưu (chạy tương tác không lưu Output). Dung lượng ~0,8 MB/mẫu (part1 ~8000 mẫu ≈ 6,5 GB tổng).
4. Mặc định: Wav2Lip GAN cho mọi split, MuseTalk 1.5 chỉ ở test; train chỉ `donor`, validation/test có `donor` + `source`; fake cục bộ có **độ dài ngẫu nhiên liên tục** trong `PARTIAL_RANGE_S` (0,2–3,0 s). Clip không có donor cùng `speaker_id` bị bỏ và ghi vào plan. Generator AI thật phải đứng đầu mỗi split (sinh kèm real/sham).
5. **MechFake** (v002, train/validation, mỗi parent đủ 5 cơ chế, tiếng gốc, không cần worker):

   | Cơ chế | Làm gì | Họ generator mô phỏng |
   |---|---|---|
   | `mech_blend` (M1) | Trộn biến thể màu / độ nét / dịch nhỏ của chính vùng miệng | Dán–ghép (SBI, SAVe) |
   | `mech_lowres` (M2) | Thu còn 30–70 % cạnh hộp miệng thật rồi phóng lại | GAN độ phân giải thấp (Wav2Lip) |
   | `mech_vae` (M3, GPU) | Qua SD-VAE ft-mse, mỗi ô latent phủ 2–5 px frame | Latent (MuseTalk, LatentSync) |
   | `mech_warp` (M4) | Uốn kết cấu miệng của khung cách ±3–25 bằng optical flow | Biến dạng (DINet) |
   | `mech_temporal` (M5) | Chỉ vùng môi: lệch ±5–25 khung / giữ khung / đổi tốc độ | Sai thời điểm (AVPF) |

   Vùng miệng từ YuNet (mép miệng, mũi), mặt nạ elip làm mềm; vùng môi hoặc nửa dưới mặt chọn ngẫu nhiên (M5 chỉ môi). Mỗi parent một cơ chế được đánh `feature_scope = full` (trích đủ AV-HuBERT để train head hợp điểm); còn lại `artifact` (chỉ DINOv2, hộp miệng lấy từ real của parent qua `box_sample_id`). Năm cơ chế của cùng parent dùng chung clip đã giải mã + điểm mặt.
6. **Chẩn đoán E6** (chỉ test, nhãn 0): `diag_paste` (nửa dưới mặt thật thu về ≤96 px như Wav2Lip rồi dán lại; ở 640 px gần như chỉ còn viền dán), `diag_global` (thu nhỏ / làm nét / gamma toàn khung rồi nén CRF 28–35). Báo riêng trong `test.json → diagnostics`, không vào chỉ số chính.
7. Thời gian mỗi phiên (giây/cặp theo generator) ghi vào `generation.json` → `sessions` (dùng cho bảng chi phí).

Nhãn mỗi mẫu: `label` (AI), `fake_intervals`, `audio_mode` (`donor`/`source`/null), `av_mismatch_intervals` (nhãn "lệch tiếng" cho nhánh sync P2 / TriCue, không phải nhãn deepfake), `diagnostic` (mẫu E6), `feature_scope`/`box_sample_id` (MechFake).

## 3. Duyệt, chốt, đóng gói (local)

Tải `vn-av-df-forensics/datasets/vn-av-df-data/<part>/` và `vn-av-df-forensics/generation/plans/<part>.json` về đúng chỗ, đặt `PART` trong `data_settings.py`, chạy trong `generation/`:

- `03b_merge.py` (khi sinh nhiều phiên độc lập): tải Output **mỗi phiên vào một thư mục riêng**, vd. `python tools/kaggle_pull.py <user>/<notebook> kaggle_outputs/gpu "/datasets/|/generation/plans/"` hoặc tải ZIP trên Kaggle rồi giải nén vào `kaggle_outputs/<phiên>/` (trong repo, đã `.gitignore`; trong mỗi thư mục phải thấy `vn-av-df-forensics/`), điền `MERGE_INPUTS` trong `settings.py` (danh sách các thư mục đó), chạy. Kiểm cùng plan, cùng chữ ký code, provenance, hash từng video; thiếu cặp thì báo theo generator và không chép gì. Ghi vào `datasets/vn-av-df-data/<part>/` (phải trống) + `generation/plans/<part>.json`, kèm `candidates.jsonl`/`review.csv`. Một Output đủ cả part (gộp trên Kaggle hoặc một notebook tuần tự) thì tải thẳng về đúng chỗ, bỏ qua bước này.
- `04_review.py`: http://127.0.0.1:8002. **Mặc định mọi mẫu `keep`**, chỉ đánh `reject`/`uncertain` mẫu lỗi. Real gốc chiếu cạnh mẫu đang duyệt; timeline vạch đoạn fake (đỏ) và đoạn ghép tiếng của sham (cam); lọc theo split/variant/generator/audio_mode.
- `05_finalize.py`: chốt mẫu keep (bất biến; đổi thì làm part mới).
- `06_export.py`: tạo `datasets/vn-av-df-data/<part>.zip` → upload thành `vn-av-df-data-partN`.

## 4. Trích feature (Kaggle, [prepare.ipynb](notebooks/prepare.ipynb))

```python
DATA_PARTS = [Path("/kaggle/input/datasets/<user>/vn-av-df-data-part1")]
ENCODERS = "all"     # avhubert + dinov2; hoặc ["avhubert"], ["dinov2"] (cần avhubert/ trong CACHE_INPUTS)
CACHE_INPUTS = []    # cache đã có: trích tiếp, hoặc lấy hộp miệng AV-HuBERT cho DINOv2
```

Đo ở v001 (part1, 2954 mẫu, 2×T4): AV-HuBERT ~4,7 h (4 worker), DINOv2 ~0,4 h. Cache AV-HuBERT v2 lưu thêm landmark môi + đường bao tiếng (nhánh sớm–muộn TriCue); mẫu MechFake `feature_scope = artifact` không trích AV-HuBERT. Output chỉ giữ `cache/vn-av-df-data/<encoder>/` đã chọn. Vì giới hạn 500 file, tạo dataset `vn-av-df-features-partN` ở local:

```powershell
python tools/kaggle_pull.py <user>/<prepare-slug> .. "/cache/"      # tải Output về repo (thư mục cha của vn-av-df-forensics)
cd cache/vn-av-df-data; tar -a -cf D:\vn-av-df-features-part1.zip avhubert dinov2   # chỉ thư mục encoder
```

Hoặc trong File Explorer: mở `cache\vn-av-df-data\`, chọn **cùng lúc `avhubert` và `dinov2`** → *Compress to ZIP*. Quy tắc ZIP:

- Tầng đầu của ZIP là **trực tiếp** `avhubert/`, `dinov2/`; không zip thư mục cha `vn-av-df-data`, không kèm `environment/` (chỉ là thông tin worker).
- `CACHE_INPUTS` trong `train.ipynb` trỏ tới **thư mục chứa trực tiếp** `avhubert/`, `dinov2/`: ZIP đúng → `Path("/kaggle/input/datasets/<user>/vn-av-df-features-part1")`; lỡ zip cả thư mục cha → thêm `/vn-av-df-data`.
- Upload thành dataset `vn-av-df-features-partN` (Kaggle tự giải nén). Prepare hai phiên (vd. AV-HuBERT, DINOv2 riêng) thì zip mỗi encoder riêng hoặc gộp đều được; liệt kê mọi dataset trong `CACHE_INPUTS`.

## 5. Train + test ([train.ipynb](notebooks/train.ipynb))

```python
DATA_PARTS = [Path("/kaggle/input/datasets/<user>/vn-av-df-data-part1")]   # trùng lúc prepare
CACHE_INPUTS = [Path("/kaggle/input/datasets/<user>/vn-av-df-features-part1")]
ARCHITECTURES = "all"   # hoặc danh sách, vd. ["avh_tcn", "tricue_mech"], ["art_all", "art_drop_m3"]
SEEDS = [42, 43, 44]
RUN_NAME = "train_part1_v002"
RUN_TEST = True         # test ngay sau train (ngưỡng/checkpoint chọn trên validation, khoá test-lock.json)
```

- Không tải encoder, không cài worker; gắn cache bằng symlink và kiểm cache đủ encoder, đúng hash dữ liệu. Output chỉ giữ `runs/<RUN_NAME>/`.
- `best.pt` mỗi detector chọn theo validation loss; báo cáo validation/test từng model/seed (theo ô 2×2, theo nhánh P2, theo generator); không tự xếp hạng. Đổi dữ liệu/model/hyperparameter thì dùng `RUN_NAME` mới.
- Tải về: `python tools/kaggle_pull.py <user>/<train-slug> .. "/runs/"`. Hai lần train cùng dữ liệu có thể gộp vào một `runs/<RUN_NAME>` (gộp `training.json`/`evaluation.json`, dựng lại `test-lock.json`) hoặc để riêng và liệt kê cả hai trong `compare.ipynb`.
- **Local (CPU):** đặt `RUN_NAME`, `ARCHITECTURES`, `DEVICE = "cpu"` trong `settings.py`, cache ở `cache/vn-av-df-data/`, rồi `python training/02_train.py` → `python training/03_evaluate.py`.

| Kiến trúc (`"all"`) | Train bằng fake | Vai trò |
|---|---|---|
| `avh_tcn` **B-AVH** | Wav2Lip | AV-HuBERT Base LRS3 (frozen) → concat A/V → TCN (≈ AV-Lip-Sync+) |
| `p2_syncartifact` **P2** | Wav2Lip | Đề xuất v001: residual A→V + head sync không thấy fake + DINOv2 crop miệng + gate |
| `tricue_mech` **TriCue** (đề xuất v002) | MechFake | Ba nhánh, hai đầu ra (xem dưới); **không cần fake của generator khi train** |
| `tricue_w2l`, `tricue_mech_w2l` | Wav2Lip / cả hai | Cùng kiến trúc, đổi dữ liệu train |
| `tricue_mech_no_lead`, `_no_sync`, `_no_art` | MechFake | Bỏ từng nhánh |
| `art_all`, `art_only_m1…m5`, `art_m1_m5`, `art_drop_m1…m5` | Một phần MechFake | Chỉ nhánh dấu vết vẽ: bản đồ cơ chế → generator |

TriCue học từng nhánh rồi hợp điểm: **m_sync** phân loại độ lệch tiếng–miệng (−K…K frame + "không khớp") trên hồ sơ cos AV-HuBERT, học từ real tự dịch lệch + sham, không thấy fake; **m_lead** khoảng cách Mahalanobis của hồ sơ tương quan chéo độ mở môi ↔ đường bao tiếng quanh điểm bắt đầu nói so với real (không train); **r_art** DINOv2 crop miệng → TCN, học real vs fake của method; **hợp điểm** logistic trọng số không âm → đầu ra "AI vẽ?" (chỉ số chính) và "lệch tiếng?" (`branches` → `mismatch`). P2 vẫn học 3 stage A/S/C như v001 để làm mốc.

## 6. So sánh ([compare.ipynb](notebooks/compare.ipynb))

Đọc `runs/` đã test, **mỗi part một bộ bảng** (+ bảng gộp khi nhiều part): video (mean ± std qua seed, seed-ensemble), theo generator, theo ô 2×2 (+ false alarm real/sham), theo nhánh, định vị thời gian, **paired ΔAUC** (bootstrap theo clip gốc, có CI), chi phí train/trích/sinh và dự tính GPU cho part mới. Ghi `reports/compare/<split>_<part>.md` + PNG. Ở local chạy bằng env `vn-av-df` (cài `ipykernel` một lần, xem đầu notebook); trên Kaggle attach Output train rồi sửa `RUNS`.

## 7. Demo (local)

1. **B-AVH/P2:** cài worker AV-HuBERT theo [environments/README.md](environments/README.md), điền `AVH_PYTHON` trong `settings.py`.
2. Đặt `runs/<RUN_NAME>/` (có `training.json`), chọn `RUN_NAME`, `DEMO_METHOD`, `SEEDS` (hoặc `CHECKPOINT` tới `best.pt`) trong `settings.py`.
3. `cd demo; npm ci; npm run build; cd ..` rồi `python training/04_demo.py`.

Demo nhận clip một người nói, tối đa 60 s: timeline AI + ngưỡng (từ validation); P2/TriCue thêm điểm/timeline từng nhánh (TriCue có thêm `lead` và đầu ra `mismatch`; chưa có ngưỡng riêng; nhánh sync cao cũng có thể chỉ do lồng tiếng thông thường). Bảng "Kết quả test" lấy từ `runs/<RUN_NAME>/evaluation.json`.

## Kiểm thử

```powershell
python -m pytest -q tests data_pipeline/tests
python -m ruff check src tests settings.py data_settings.py data_pipeline/src data_pipeline/tests data_pipeline/steps tools
```

Test dùng generator giả và feature fixture, nên **không chứng minh checkpoint thật chạy được hay detector chính xác**.

## Phiên bản và checkpoint được chọn

Bảng mô tả tài sản được cấu hình sử dụng; không có nghĩa tất cả đã có ở local. Checkpoint generator được tải trên Kaggle.

| Thành phần | Phiên bản/checkpoint được chọn |
|---|---|
| YuNet | Bản March 2023 của OpenCV Zoo: `face_detection_yunet_2023mar.onnx` |
| Silero VAD | **v6.0**: `silero_vad.onnx` |
| AV-HuBERT | **Base, LRS3, clean-pretrain, iteration 4**, chưa fine-tune: `base_lrs3_iter4.pt` |
| Dlib landmark cho AV-HuBERT | Landmark 68 điểm: `shape_predictor_68_face_landmarks.dat`, tải dạng `.dat.bz2` rồi giải nén |
| Mean-face cho AV-HuBERT | `20words_mean_face.npy` từ `mpc001/Lipreading_using_Temporal_Convolutional_Networks`; dữ liệu tham chiếu tiền xử lý |
| DINOv2 cho nhánh dấu vết vẽ (P2, TriCue) | `facebook/dinov2-small` (ViT-S/14), `model.safetensors`; revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| VAE của MechFake (`mech_vae`) | `stabilityai/sd-vae-ft-mse`: `diffusion_pytorch_model.safetensors`, revision ghim lần tải đầu (`weights/sd-vae-ft-mse/vn_av_revision.json`) |
| YuNet cho MechFake | Như dòng YuNet, tải về `weights/media/yunet.onnx` |
| Wav2Lip | **Wav2Lip GAN**, lưu thành `Wav2Lip-SD-GAN.pt` |
| Face detector của Wav2Lip | **S3FD**, nguồn `s3fd-619a316812.pth`, lưu thành `s3fd.pth` |
| MuseTalk | **MuseTalk 1.5**, repository `TMElyralab/MuseTalk`: `musetalkV15/unet.pth` và cấu hình `musetalkV15/musetalk.json` |
| VAE của MuseTalk | `stabilityai/sd-vae-ft-mse`: `diffusion_pytorch_model.bin` |
| Audio encoder của MuseTalk | `openai/whisper-tiny`: `pytorch_model.bin` |
| DWPose của MuseTalk | `yzd-v/DWPose`: `dw-ll_ucoco_384.pth` |
| Face parsing của MuseTalk | `79999_iter.pth` và `resnet18-5c106cde.pth` |
| Detector của project | `best.pt` riêng cho từng kiến trúc/seed, tạo sau khi train; demo dùng kèm encoder tương ứng |

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/vn_av_df/` | Generation, feature, model, train/test, demo |
| `data_pipeline/` | Thu thập, cắt, duyệt, export + split (`steps/01…05`) |
| `generation/`, `training/` | Entrypoint local (`00…06` + `03b_merge`, `00…04`) |
| `notebooks/` | `cut`, `generate`, `prepare`, `train` (Kaggle) và `compare` |
| `tools/` | `kaggle_pull.py` |
| `environments/` | Worker MuseTalk / AV-HuBERT |
| `demo/` | Frontend |
| `tests/`, `data_pipeline/tests/` | Kiểm thử |

Không commit weights, external, cache, datasets, runs, reports. Nhật ký nghiên cứu nằm trong `RESEARCH_WORKLOG`.
