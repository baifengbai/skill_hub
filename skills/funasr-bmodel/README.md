# FunASR-bmodel — 中文语音识别

基于 [wlc952/FunASR-bmodel](https://modelscope.cn/models/wlc952/FunASR-bmodel)，在 BM1684X TPU 上运行 Paraformer Large 模型，提供 Web 界面和 API。

## 功能

- 🎙 浏览器麦克风录音 → 实时识别
- 📂 上传 WAV / MP3 / M4A 文件识别
- 🔤 中文 ASR + VAD 断句 + 标点恢复
- ⚡ 推理在 TPU 上执行，系统内存占用低

## API

```
POST /api/recognize
Content-Type: multipart/form-data
Body: audio=<file>

Response: {"text": "识别结果", "inference_time": 1.23}
```

## 部署

见 [SKILL.md](./SKILL.md)。

## 文件

| 文件 | 说明 |
|------|------|
| `SKILL.md` | 完整部署文档 |
| `funasr_web.py` | Flask Web 服务（含前端 HTML） |
