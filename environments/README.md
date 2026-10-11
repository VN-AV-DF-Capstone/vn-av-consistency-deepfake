# Môi trường worker

**Generation và training/evaluation chạy trên Kaggle.** Đây là cách tách dependency, không phải mỗi baseline cần một môi trường và không yêu cầu cài toàn bộ lên local.

| Phiên chạy | Môi trường chính | Worker riêng cần tạo trong phiên đó |
|---|---|---|
| Kaggle generation | Package profile generation | MuseTalk/Python 3.10 chỉ khi chọn musetalk_1_5. |
| Kaggle training | Profile training: các detector đã chọn | AV-HuBERT/Python 3.10 chỉ khi có model dùng encoder AV-HuBERT. |
| Local review | Package media/review/finalize | Không cần worker generator/encoder. |
| Local demo video mới | Inference detector checkpoints | AV-HuBERT cho B-AVH/P2/TriCue; không cần MuseTalk. |

## Tự chuẩn bị worker trên Kaggle

Hai notebook có cell gọi [bootstrap_worker.py](bootstrap_worker.py) sau bước tải assets. Script tải Micromamba từ endpoint chính thức, tạo Python 3.10 ở `--env-root` (notebook: `WORKERS = /kaggle/temp/vn-av-df-workers`, ngoài /kaggle/working nên không tính vào Output 20 GB) và cài đúng worker; không yêu cầu Kaggle có conda từ trước và không cài fairseq/MMLab vào kernel chính. Notebook tự gán đường dẫn interpreter vào config sau khi setup đạt.

| Notebook | Worker tự tạo | Không cài trong bước worker |
|---|---|---|
| generate.ipynb | WORKERS/.venv-musetalk/bin/python nếu chọn MuseTalk | AV-HuBERT |
| prepare.ipynb | WORKERS/.venv-avhubert/bin/python nếu `ENCODERS` có `avhubert` | MuseTalk/Wav2Lip |

Bật Internet và GPU rồi chạy các cell theo thứ tự. Bootstrap tự kiểm Python, phiên bản torch, tương thích NumPy, import thư viện và phép tính CUDA nhỏ. Receipt chỉ đánh dấu ready sau khi pip check và import/CUDA probe đạt. Đây **không phải** inference thử checkpoint thật.

Các kiểm thử orchestration local đã đạt, nhưng **chưa có phiên Kaggle end-to-end hoặc kiểm chứng checkpoint thật**. Nếu pip/conda/import lỗi, cell dừng trước generation/feature extraction và chỉ rõ file log; không thay model hoặc tự bỏ qua worker.

Log và thông tin môi trường nằm ở `<env-root>/.kaggle-tools/<worker>/` (mặc định env-root = project root): setup.log, environment.json, constraints.txt, probe.json, pip-freeze.txt. Chạy lại cùng cấu hình có thể tiếp tục phần cài còn dở; môi trường ready được kiểm lại và không tự cài lại. Đổi YAML/source/bootstrap phải dùng checkout làm việc mới để tránh trộn môi trường. Script không sửa một prefix có sẵn nhưng không có receipt do nó tạo. pip chạy với `PIP_NO_CACHE_DIR=1`; cài xong thì xoá cache gói micromamba (`.kaggle-tools/mamba/pkgs`). Môi trường ở /kaggle/temp mất khi hết phiên: phiên mới cài lại.

Notebook generation xuất thêm generation-environment.zip; notebook training đưa thông tin worker vào ZIP kết quả. Micromamba tải lần đầu từ endpoint latest, lưu URL cuối và hash binary/archive; đây là pin tài sản đã tải, không phải khóa sẵn mọi dependency transitive. Nguồn: [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), [wheel MMCV CUDA11.8/torch2.0](https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html).

Có thể gọi trực tiếp từ project root trên Linux x86_64 sau khi tải source/weights:

```bash
python environments/bootstrap_worker.py musetalk
# Trong phiên training riêng:
python environments/bootstrap_worker.py avhubert
```

`--allow-cpu` chỉ bỏ điều kiện phải có GPU trong probe; không bảo đảm mọi thao tác upstream chạy được trên CPU. Bootstrap tự động không chạy trên Windows, tránh cài nhầm worker Kaggle lên máy local.

## Cài thủ công khi cần

Các lệnh dưới dành cho môi trường đã có conda, ví dụ local demo P1. Không cần chạy lại nếu cell bootstrap Kaggle đã đạt.

Trong notebook generation, action generator_setup tải source/weights generator; notebook training dùng encoder_setup cho encoder. Đó là các action tương ứng `generation/00_setup.py` và `training/00_setup.py`, không yêu cầu chạy cả hai trên local. Cần Internet; tải source xong mới cài dependency tham chiếu external/ của worker tương ứng.

AV-HuBERT - dùng trong Kaggle training, hoặc local demo P1 (từ project root):

```powershell
# Trong môi trường Anaconda chính: cài thư viện demo và tự tải source/weights trước.
python 00_install.py
python training/00_setup.py --encoder avhubert

# Tạo worker riêng, không cài fairseq vào môi trường chính.
conda env create -f environments/avhubert.yml
conda run -n vn-avhubert python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
conda run -n vn-avhubert python -m pip install --no-build-isolation -e external/av_hubert/fairseq
conda run -n vn-avhubert python -m pip check
conda run -n vn-avhubert python -c "import sys; print(sys.executable)"
```

Copy đường dẫn Python in ở lệnh cuối vào `AVH_PYTHON` trong settings.py. Tải assets tự động không có nghĩa worker đã được cài; Windows/fairseq vẫn cần kiểm chứng bằng inference thực tế. Bootstrap Micromamba trong notebook dành cho Kaggle/Linux, không chạy nó để cài worker Windows.

AV-HuBERT dùng fairseq submodule của source đã khóa. Python 3.10 tránh lỗi dataclass của fairseq cũ trên Python mới; pip <24.1 đọc được metadata OmegaConf cũ. Torch 2.0.1 là lựa chọn tương thích ban đầu, chưa phải môi trường được xác nhận bằng media thật. [Hướng dẫn upstream](https://github.com/facebookresearch/av_hubert#installation).

MuseTalk - dùng trong Kaggle generation:

```powershell
conda env create -f environments/musetalk.yml
conda run -n vn-musetalk python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
conda run -n vn-musetalk python -m pip install -r external/MuseTalk/requirements.txt
conda run -n vn-musetalk python -m pip install openmim
conda run -n vn-musetalk mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"
conda run -n vn-musetalk python -m pip check
```

Đây là phiên bản torch/MMLab theo [hướng dẫn MuseTalk chính thức](https://github.com/TMElyralab/MuseTalk#installation). Windows có thể cần compiler nếu không có wheel phù hợp; Linux/CUDA 11.8 là lựa chọn thực thi ban đầu. Không thay checkpoint khi cài lỗi.

Lấy đường dẫn Python bằng `conda run -n vn-avhubert python -c "import sys; print(sys.executable)"` (tương tự MuseTalk), điền `AVH_PYTHON` và `MUSETALK_PYTHON` trong `settings.py`, hoặc đặt `VN_AVH_PYTHON`, `VN_MUSETALK_PYTHON` **trước** khi import settings. Notebook Kaggle cũng cần các interpreter này; kernel chính không tự trở thành worker.

Trước benchmark: chạy generate 2 clip/split, review, prepare cả hai encoder, train 1 seed/2 epoch, evaluate và demo; kiểm tra hash, số frame, coverage, loss tái dựng so với constant predictor, VRAM, thời gian. Ghi `pip freeze`, `conda list --explicit`, GPU/CUDA và revision vào thư mục run. Unit tests dùng fixture không thay thế bước này.
