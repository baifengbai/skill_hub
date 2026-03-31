# YOLOv5-bmodel — 目标检测

基于 sophon-demo YOLOv5s，在 BM1684X TPU 上检测 COCO 80 类目标，提供 Web 界面和 REST API。

## 功能

- 🖼️ 图片上传 / 服务器图库选择 / Base64 输入
- 🎯 80类 COCO 目标检测，可视化标注框 + 标签
- ⚙️ 实时调节置信度/NMS阈值
- ⚡ sophon.sail TPU 推理，单张约 50ms

## API

```
POST /api/detect
Body: file=<图片> | path=<路径> | image_base64=<base64>
      conf_thresh=0.25  nms_thresh=0.45

Response: {"detections":[{"label":"...","conf":0.87,"box":[x1,y1,x2,y2]}], "count":N, "time_ms":58, "image":"data:..."}
```

## 部署

见 [SKILL.md](./SKILL.md)。

## 文件

| 文件 | 说明 |
|------|------|
| `SKILL.md` | 完整部署文档 |
| `yolov5_app.py` | Flask Web 服务（含前端 HTML + API 文档页） |
