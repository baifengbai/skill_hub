# ByteTrack-bmodel — 多目标追踪

基于 sophon-demo YOLOv5s + ByteTrack，在 BM1684X TPU 上实现多目标追踪，提供 Web 界面和 REST API。

## 功能

- 🎬 视频上传，逐帧检测 + 多目标追踪
- 🆔 每个目标分配唯一 Track ID，跨帧关联
- 📊 自动选择 bmcv/TPU 或 OpenCV/CPU 后端
- ⚡ TPU 加速推理

## API

```
POST /api/track
Body: video=<视频文件>

Response: SSE 流式返回追踪结果
```

## 部署

见 [SKILL.md](./SKILL.md)。

## 文件

| 文件 | 说明 |
|------|------|
| `SKILL.md` | 完整部署文档 |
| `bytetrack_web.py` | Flask Web 服务（含前端 HTML） |
