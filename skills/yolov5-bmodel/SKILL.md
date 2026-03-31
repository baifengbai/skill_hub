---
name: yolov5-bmodel
description: 在 SOPHON BM1684X 设备上部署 YOLOv5 目标检测服务，含 Web 前端和 REST API，通过 gssh 端口转发本地访问。
allowed-tools: Read, Write, Glob, Grep, Bash
version: 1.0.0
---

## 概述

在 SOPHON BM1684X 上运行 YOLOv5s（COCO 80类目标检测），使用 `sophon.sail` 加速推理，Flask 提供 Web + API 服务。

**模型来源：** sophon-demo 已内置，无需额外下载。

---

## 环境要求

- SOPHON BM1684X SOC，Ubuntu 20.04 aarch64
- 系统 Python 3.8 + `sophon.sail`（系统已安装）
- Flask（`pip3 install flask`）
- OpenCV（`pip3 install opencv-python-headless`）

---

## 模型路径

| 精度 | 路径 |
|------|------|
| fp16（推荐） | `/data/sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel` |
| fp32 | `/data/sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_fp32_1b.bmodel` |
| int8 1b | `/data/sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel` |
| int8 4b | `/data/sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel` |

依赖的 Python 辅助文件（后处理 + 颜色/类别表）：
```
/data/sophon-demo/sample/YOLOv5/python/postprocess_numpy.py
/data/sophon-demo/sample/YOLOv5/python/utils.py
```

---

## 部署步骤

### 1. 上传服务代码

将 `yolov5_app.py` 上传到远程机器：

```bash
gssh scp -put yolov5_app.py /home/<USERNAME>/yolov5_app.py
```

### 2. 后台启动服务

```bash
# 系统 Python 3.8 + sophon.sail，可直接运行
gssh exec "nohup python3 /home/<USERNAME>/yolov5_app.py > /tmp/yolov5.log 2>&1 &"

# 查看启动日志
gssh exec "cat /tmp/yolov5.log"
```

服务默认端口：**15001**

### 3. 本地访问

```bash
gssh forward -l 15001 -r 15001
# 浏览器打开 http://localhost:15001
# API 文档：http://localhost:15001/api-doc
```

---

## API 接口

### POST /api/detect

检测图片中的目标，支持三种输入方式：

| 字段 | 类型 | 说明 |
|------|------|------|
| `file` | file | multipart/form-data 上传图片 |
| `path` | string | 服务器绝对路径 |
| `image_base64` | string | Base64 编码图片 |
| `conf_thresh` | float | 置信度阈值，默认 0.25 |
| `nms_thresh` | float | NMS IOU 阈值，默认 0.45 |

**响应：**
```json
{
  "detections": [
    {"label": "person", "conf": 0.872, "box": [120.0, 45.0, 380.0, 610.0]}
  ],
  "count": 1,
  "time_ms": 58,
  "image": "data:image/jpeg;base64,..."
}
```

**cURL 示例：**
```bash
curl -X POST http://localhost:15001/api/detect \
  -F "file=@image.jpg" \
  -F "conf_thresh=0.25"
```

**Python 示例：**
```python
import requests

resp = requests.post(
    "http://localhost:15001/api/detect",
    files={"file": open("image.jpg", "rb")},
    data={"conf_thresh": 0.25, "nms_thresh": 0.45}
)
data = resp.json()
for d in data["detections"]:
    print(d["label"], d["conf"], d["box"])
```

### GET /api/images

返回服务器内置测试图片列表（ImageNet val 1k）。

### GET /api/thumb?p={路径}

返回 120px 宽缩略图（JPEG）。

### GET /api/preview?p={路径}

返回最大边 800px 的预览图（JPEG）。

---

## 关键技术点

- 使用 `sophon.sail.Engine` 加载 bmodel，推理在 TPU 上执行
- letterbox 预处理保持纵横比（填充灰边）
- 后处理依赖 `postprocess_numpy.PostProcess`（NMS 在 CPU 完成）
- 动态调整 conf/nms 阈值：只重建 PostProcess，不重新加载 bmodel

---

## 关键路径

| 内容 | 路径 |
|------|------|
| Web 服务 | `/home/<USERNAME>/yolov5_app.py` |
| 服务日志 | `/tmp/yolov5.log` |
| bmodel（fp16）| `/data/sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel` |
| 后处理脚本 | `/data/sophon-demo/sample/YOLOv5/python/` |
| 测试图片 | `/data/sophon-demo/sample/ResNet/datasets/imagenet_val_1k/img/` |
