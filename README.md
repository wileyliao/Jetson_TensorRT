# Jetson_TensorRT
Computer Vision on edge device guid, use yolo as example

## 1. 環境需求
- Jetson Nano / Xavier NX / Orin 系列
- JetPack (含 CUDA, cuDNN, TensorRT)
- Python 3.8+
```
sudo apt-get install libzbar0
pip install ultralytics opencv-python
```
## 2. 模型匯出
將 YOLO .pt 權重轉成 TensorRT engine：
```python
from ultralytics import YOLO

# 匯出成 TensorRT engine (FP16)
model = YOLO("qr_bar_code.pt")
model.export(format="engine", half=True, imgsz=640)
# 會輸出 qr_bar_code.engine
```

## 3. 推論範例
原本code:
```python
model = YOLO("qr_bar_code.pt")
yret = model(img)[0]
```
TensorRT code:
```python
model = YOLO("qr_bar_code.engine")
yret = model(img)[0]   # 前/後處理流程可共用
```

## 4. INT8 量化
若要進一步壓縮，需提供校正集以維持精度。
```python
model.export(format="engine", int8=True, imgsz=640, data="calib_images/")
```

## 5. 執行環境
### Jetson 上裸跑
- 方式：直接在 Jetson 裝 CUDA、cuDNN、TensorRT（JetPack）- pip install PyTorch / Ultralytics。
#### 優點：
- 少一層容器，效能最佳。
- Debug / 使用攝影機（CSI、USB、GStreamer）比較直覺。

#### 缺點：
- 軟體版本一亂就容易踩坑（PyTorch / TorchVision / TensorRT 版本不合）。
- 不同專案之間環境難隔離。

### Jetson 上 Docker
- 方式：用 NVIDIA 官方提供的 L4T Docker image，內建 CUDA / cuDNN / TensorRT，跑在 nvidia-docker 上
```bash
sudo docker run --runtime nvidia -it --rm \
    --network host \
    --volume ~/workspace:/workspace \
    nvcr.io/nvidia/l4t-ml:r35.2.1-py3

```
- `--runtime nvidia` → 讓 Docker 有 GPU 權限
- `l4t-ml` → NVIDIA 官方 ML 容器（已含 PyTorch、TensorRT）
#### 優點：
- NVIDIA 官方 image 不用頭痛環境
- 各專案隔離，部署一致性

#### 缺點：
- Debug 寫 USB/GIGE 攝影機、GPIO、串口時，有時要額外加 --device
- 映像檔大，第一次拉要久
