---
name: ppocr-bmodel
description: 在 SOPHON BM1684X 设备上部署 PP-OCRv4 文字识别服务，含 Web 前端和 REST API，通过 gssh 端口转发本地访问。
allowed-tools: Read, Write, Glob, Grep, Bash
version: 1.0.0
---

## 概述

在 SOPHON BM1684X 上运行 PaddleOCR v4（检测 + 识别两阶段），使用 `sophon.sail` 加速推理，Flask 提供 Web + API 服务，支持中文文字检测与识别。

**模型来源：** sophon-demo 已内置，无需额外下载。

---

## 环境要求

- SOPHON BM1684X SOC，Ubuntu 20.04 aarch64
- 系统 Python 3.8 + `sophon.sail`（系统已安装）
- Flask、Pillow（`pip3 install flask pillow`）
- OpenCV（`pip3 install opencv-python-headless`）

---

## 模型路径

| 模型 | 路径 |
|------|------|
| 检测 fp32 | `/data/sophon-demo/sample/PP-OCR/models/BM1684X/ch_PP-OCRv4_det_fp32.bmodel` |
| 检测 fp16 | `/data/sophon-demo/sample/PP-OCR/models/BM1684X/ch_PP-OCRv4_det_fp16.bmodel` |
| 识别 fp32 | `/data/sophon-demo/sample/PP-OCR/models/BM1684X/ch_PP-OCRv4_rec_fp32.bmodel` |
| 识别 fp16 | `/data/sophon-demo/sample/PP-OCR/models/BM1684X/ch_PP-OCRv4_rec_fp16.bmodel` |

依赖文件：
```
/data/sophon-demo/sample/PP-OCR/python/ppocr_det_opencv.py   # 检测模块
/data/sophon-demo/sample/PP-OCR/python/ppocr_rec_opencv.py   # 识别模块
/data/sophon-demo/sample/PP-OCR/python/ppocr_cls_opencv.py   # 方向分类（可选）
/data/sophon-demo/sample/PP-OCR/datasets/ppocr_keys_v1.txt   # 字典文件
/data/sophon-demo/sample/PP-OCR/datasets/fonts/simfang.ttf   # 可视化字体
```

---

## 部署步骤

### 1. 安装依赖

```bash
pip3 install flask pillow
```

### 2. 上传服务代码

```bash
gssh scp -put ppocr_app.py /data/sophon-demo/sample/PP-OCR/python/app.py
```

### 3. 后台启动服务

```bash
# 必须在 PP-OCR python 目录下启动（内部 import 依赖相对路径）
gssh exec "nohup python3 /data/sophon-demo/sample/PP-OCR/python/app.py > /tmp/ppocr.log 2>&1 &"

# 查看启动日志
gssh exec "cat /tmp/ppocr.log"
# 看到 "Models loaded." 表示成功
```

服务默认端口：**8899**

### 4. 本地访问

```bash
gssh forward -l 8899 -r 8899
# 浏览器打开 http://localhost:8899
```

---

## API 接口

### POST /ocr

上传图片，返回检测到的文字块、置信度及可视化结果图。

**请求（multipart/form-data）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | file | 待识别图片（JPG/PNG/BMP/WEBP） |

**响应：**
```json
{
  "count": 3,
  "texts": ["发票", "金额", "¥128.00"],
  "scores": [0.98, 0.96, 0.94],
  "time_ms": 120,
  "original_b64": "...",
  "result_b64": "..."
}
```

**cURL 示例：**
```bash
curl -X POST http://localhost:8899/ocr \
  -F "image=@invoice.jpg"
```

**Python 示例：**
```python
import requests, base64

resp = requests.post(
    "http://localhost:8899/ocr",
    files={"image": open("image.jpg", "rb")}
)
data = resp.json()
print(f"识别到 {data['count']} 个文字块，耗时 {data['time_ms']}ms")
for text, score in zip(data["texts"], data["scores"]):
    print(f"  {text}  ({score*100:.1f}%)")
```

---

## 关键技术点

### 两阶段推理流程

1. **检测（Det）**：`PPOCRv2Det` 找出图中所有文字区域（输出四边形框）
2. **矫正裁剪**：`get_rotate_crop_image` 对每个文字框做透视变换，裁剪出文字图块
3. **识别（Rec）**：`PPOCRv2Rec` 批量识别每个文字图块，输出文字+置信度

### 重要配置参数

```python
class OCRArgs:
    bmodel_det = '.../ch_PP-OCRv4_det_fp32.bmodel'
    bmodel_rec = '.../ch_PP-OCRv4_rec_fp32.bmodel'
    det_limit_side_len = [640]        # 检测时图片最长边缩放到 640
    img_size = [[320, 48], [640, 48]] # 识别时图片 resize 目标尺寸
    rec_thresh = 0.5                  # 识别置信度过滤阈值
    use_angle_cls = False             # 不使用方向分类（可开启提升斜体识别）
    char_dict_path = '.../ppocr_keys_v1.txt'  # 中文字典
```

### 启动目录要求

`app.py` 内部执行 `os.chdir('/data/sophon-demo/sample/PP-OCR/python')`，确保相对 import 正常。若更改部署路径，需相应修改此行。

---

## 关键路径

| 内容 | 路径 |
|------|------|
| Web 服务 | `/data/sophon-demo/sample/PP-OCR/python/app.py` |
| 服务日志 | `/tmp/ppocr.log` |
| 检测 bmodel | `/data/sophon-demo/sample/PP-OCR/models/BM1684X/ch_PP-OCRv4_det_fp32.bmodel` |
| 识别 bmodel | `/data/sophon-demo/sample/PP-OCR/models/BM1684X/ch_PP-OCRv4_rec_fp32.bmodel` |
| 字典文件 | `/data/sophon-demo/sample/PP-OCR/datasets/ppocr_keys_v1.txt` |
| 字体文件 | `/data/sophon-demo/sample/PP-OCR/datasets/fonts/simfang.ttf` |
