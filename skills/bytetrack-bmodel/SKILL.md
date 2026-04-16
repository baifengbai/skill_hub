# ByteTrack-bmodel Skill

在 SOPHON BM1684X 设备上部署 ByteTrack 多目标追踪服务，含 Web 前端展示和推理性能统计，通过 gssh 端口转发在本地浏览器访问。

**后端自动选择：** 启动时自动探测 `sophon.sail` 可用性：
- 可用：`sail.Decoder` 硬件解码视频 → `yolov5_opencv` 推理
- 不可用：`cv2.VideoCapture`（ffmpeg 软解码）→ `yolov5_opencv` 推理

> **重要**：推理始终使用 `yolov5_opencv`（3-output bmodel）。`yolov5_fuse_bmcv` 只支持 1-output fused 模型，与设备上现有 bmodel 不兼容，不可用。bmcv 仅用于视频解码加速。

## 硬件要求

- SOPHON BM1684X SOC (aarch64, Ubuntu 20.04)
- TPU 内存: 13.5 GB
- 系统内存: 建议开启 swap（防止 OOM）

## 模型路径（远程设备）

```
/data/sophon-demo/sample/ByteTrack/models/BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel
/data/sophon-demo/sample/ByteTrack/python/configs/bytetrack.yaml
/data/sophon-demo/sample/ByteTrack/datasets/test_car_person_1080P.mp4  # 内置演示视频
```

## Python 环境

使用系统 Python 3.8（`/usr/bin/python3`），它已包含：
- `sophon.sail` — TPU 推理（bmcv 路径）
- `cv2` (4.13.0) — 视频解码/绘图（opencv 路径 & 元数据获取）
- `flask` (2.2.2) — Web 服务
- `numpy`

> **注意**：本 skill 使用系统 Python 3.8 即可（依赖 `cv2`）。如果使用 Python 3.10 虚拟环境，需确保其中安装了 `opencv-python`。

## 部署步骤

### 1. 上传 Web 服务脚本

```bash
gssh scp -put bytetrack_web.py /home/linaro/bytetrack_web.py
```

### 2. 启动服务

```bash
gssh exec "nohup python3 /home/linaro/bytetrack_web.py > /data/bytetrack_web.log 2>&1 &"
```

### 3. 检查日志

```bash
gssh exec "tail -10 /data/bytetrack_web.log"
# 正常输出: Running on http://0.0.0.0:5002
```

### 4. 建立本地端口转发

```bash
gssh forward -l 5002 -r 5002
```

### 5. 浏览器访问

打开 `http://localhost:5002`，可以：
- 点击「运行内置演示视频」直接体验
- 上传自定义视频进行追踪
- 推理完成后自动批量缓存所有帧（12帧/批并行），缓存完成后流畅播放
- 支持 seek 拖动、暂停/继续
- 查看推理性能统计（FPS、检测耗时、追踪耗时）
- 下载追踪结果视频
- 历史记录面板：每次追踪结果自动保存，可随时重新播放（帧数据保留在设备 `/data` 上）

## API 说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | Web 前端页面 |
| GET  | `/api/info` | 返回当前后端类型 `{use_bmcv, backend}` |
| POST | `/api/track` | 提交追踪任务（form data） |
| GET  | `/api/status/<task_id>` | 查询任务进度与结果 |
| GET  | `/api/frame/<task_id>/<n>` | 获取第 n 帧预览 JPEG |
| GET  | `/api/result/<task_id>` | 下载结果视频 |
| GET  | `/api/history` | 获取历史记录列表（最多 50 条） |
| GET  | `/api/history/<task_id>` | 获取单条历史任务元数据 |

### POST /api/track 参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `demo` | form field | — | 设为 `1` 使用内置演示视频 |
| `video` | file | — | 上传的视频文件 |
| `conf_thresh` | float | 0.40 | 检测置信度阈值 |
| `nms_thresh` | float | 0.70 | NMS 阈值 |
| `track_thresh` | float | 0.70 | 跟踪阈值 |

### 示例

```bash
# 使用演示视频
curl -X POST http://localhost:5002/api/track -F 'demo=1'
# 返回: {"task_id": "b6e34129"}

# 查询状态
curl http://localhost:5002/api/status/b6e34129
# 返回: {"status": "running", "progress": 42, ...}
# 完成后: {"status": "done", "stats": {"avg_fps": 5.8, "frames": 592, ...}, ...}
```

## 推理性能（fp16 精度）

在 BM1684X 上的实测数据（1080P 视频）：

| 指标 | 数值 |
|------|------|
| 平均 FPS | ~5.8 |
| 平均检测耗时 | ~167 ms/帧 |
| 平均跟踪耗时 | ~5.7 ms/帧 |
| 测试帧数 | 592 帧 |
| 总 Track IDs | 41 |

## 关键代码说明

### 后端自动选择

```python
USE_BMCV = False
try:
    import sophon.sail as sail
    sail.Handle(0)           # 探测 TPU 是否可用
    USE_BMCV = True
except Exception:
    pass

from yolov5_opencv import YOLOv5  # 推理始终用 opencv 版本
```

### 检测结果格式（yolov5_opencv）

```
[x1, y1, x2, y2, conf, cls]
```

### YOLOv5 构造方式

```python
from types import SimpleNamespace
det_args = SimpleNamespace(bmodel=BMODEL, dev_id=0, conf_thresh=0.4, nms_thresh=0.7)
detector = YOLOv5(det_args)
```

### bmcv 解码路径核心流程

```python
handle      = sail.Handle(0)
bmcv_handle = sail.Bmcv(handle)
decoder     = sail.Decoder(video_path, True, 0)
while True:
    bmimg = sail.BMImage()
    if decoder.read(handle, bmimg) != 0:
        break
    # 必须转为 BGR_PACKED，再 asmat() 取 numpy，否则颜色可能错误
    bgr   = bmcv_handle.convert_format(bmimg, sail.Format.FORMAT_BGR_PACKED)
    frame = bgr.asmat()              # numpy BGR，传入 yolov5_opencv
    results = detector([frame])
```

### ByteTracker 构造方式

实际签名（非 cfg 对象）：

```python
ByteTracker(min_box_area, track_thresh, track_buffer, match_thresh)
```

使用 yaml 配置中的默认值：

```python
cfg = get_config()
cfg.merge_from_file(CFG_FILE)
tracker = ByteTracker(
    cfg.BYTETRACK.MIN_BOX_AREA,   # 10
    track_thresh,                  # 0.7
    cfg.BYTETRACK.TRACK_BUFFER,   # 30
    cfg.BYTETRACK.MATCH_THRESH    # 0.8
)
```

## 停止服务

```bash
gssh exec "pkill -f bytetrack_web.py"
```

## 数据文件说明

| 路径 | 说明 |
|------|------|
| `/data/bytetrack_frames/<task_id>/` | 追踪结果预览帧（JPEG，960×540） |
| `/data/bytetrack_output/<task_id>.mp4` | 追踪结果完整视频（mp4v） |
| `/data/bytetrack_uploads/` | 上传的原始视频临时存储 |
| `/data/bytetrack_history.json` | 历史记录（最多 50 条，持久化） |
| `/data/bytetrack_web.log` | 服务运行日志 |

> 帧数据和历史记录均在 `/data`（44GB 分区），不占用根分区空间。
