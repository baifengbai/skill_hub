# PP-OCR-bmodel — 文字识别

基于 sophon-demo PP-OCRv4，在 BM1684X TPU 上进行中文文字检测与识别，提供 Web 界面和 REST API。

## 功能

- 🖼️ 图片上传（JPG/PNG/BMP/WEBP）
- 🔤 中文文字检测 + 识别（两阶段流水线）
- 📊 可视化：原图 + 标注框 + 识别文字并排展示
- ⚡ sophon.sail TPU 推理

## API

```
POST /ocr
Body: image=<图片文件>

Response: {"count":3, "texts":["..."], "scores":[0.98], "time_ms":120, "original_b64":"...", "result_b64":"..."}
```

## 部署

见 [SKILL.md](./SKILL.md)。

## 文件

| 文件 | 说明 |
|------|------|
| `SKILL.md` | 完整部署文档 |
| `ppocr_app.py` | Flask Web 服务（含前端 HTML） |
