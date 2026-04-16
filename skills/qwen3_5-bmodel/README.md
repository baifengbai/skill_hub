# Qwen3.5-bmodel — 多模态 VL 对话

基于 Qwen3.5-VL-2B int4 AutoRound W4BF16，在 BM1684X TPU 上运行多模态视觉语言模型，支持图片、视频和纯文字问答，提供 Web 界面和 SSE 流式 API。

## 功能

- 🖼️ 图片上传 + 文字提问
- 🎬 视频上传 + 文字提问（1fps 抽帧）
- 💬 纯文字对话
- 📊 SSE 流式输出，实时显示 prefill 耗时和 tps
- ⚡ TPU 推理，seq2048，约 5-10 tok/s

## 演示

<video src="demo.mp4" controls width="720"></video>

## API

```
POST /api/chat
Content-Type: multipart/form-data
Body: message=<问题> [media=<图片/视频>]

Response: SSE 流式
  data:{"type":"perf","prefill":1.23}
  data:"累积输出..."
  data:{"type":"perf","total":5.6,"tokens":42,"tps":8.5}
  data:"__done__"
```

## 部署

见 [SKILL.md](./SKILL.md)。

## 文件

| 文件 | 说明 |
|------|------|
| `SKILL.md` | 完整部署文档（含 transformers 版本、processor_config 修复等） |
| `qwen3_5_web.py` | Flask Web 服务（含前端 HTML） |
| `demo.mp4` | 演示视频 |
