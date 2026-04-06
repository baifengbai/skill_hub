# ByteTrack-bmodel Skill

在 SOPHON BM1684X 设备上部署 ByteTrack 多目标跟踪服务，含 Web 前端展示和推理性能统计，通过 gssh 端口转发在本地浏览器访问。

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
- `sophon.sail` — TPU 推理
- `cv2` (4.13.0) — 视频处理
- `flask` (2.2.2) — Web 服务
- `numpy`

> **注意**：不要使用 `/data/AIGC-SDK/hub_venv`，该环境缺少 `cv2`。

## 部署步骤

### 1. 上传 Web 服务脚本

```bash
gssh scp -put bytetrack_web.py /home/linaro/bytetrack_web.py
```

### 2. 启动服务

```bash
gssh exec "nohup python3 /home/linaro/bytetrack_web.py > /tmp/bytetrack.log 2>&1 &"
```

### 3. 检查日志

```bash
gssh exec "tail -10 /tmp/bytetrack.log"
# 正常输出: Running on http://0.0.0.0:5002
```

### 4. 建立本地端口转发

```bash
gssh forward -l 5002 -r 5002
```

### 5. 浏览器访问

打开 `http://localhost:5002`，可以：
- 点击「运行内置演示视频」直接体验
- 上传自定义视频进行跟踪
- 查看推理性能统计（FPS、检测耗时、跟踪耗时）
- 下载跟踪结果视频

## API 说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | Web 前端页面 |
| POST | `/api/track` | 提交跟踪任务（form data） |
| GET  | `/api/status/<task_id>` | 查询任务进度 |
| GET  | `/api/result/<task_id>` | 下载结果视频 |

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

### YOLOv5 构造方式

`YOLOv5` 接受一个 args 对象而非关键字参数：

```python
from types import SimpleNamespace
det_args = SimpleNamespace(bmodel=BMODEL, dev_id=0,
                           conf_thresh=conf_thresh, nms_thresh=nms_thresh)
detector = YOLOv5(det_args)
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
