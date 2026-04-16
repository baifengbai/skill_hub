# Qwen3-bmodel — 对话 LLM

基于 Qwen3-4B AWQ W4BF16，在 BM1684X TPU 上运行文本对话模型，提供 Web 界面和 SSE 流式 API。

## 功能

- 💬 纯文字对话，SSE 流式输出
- 🧠 思考模式切换（默认关闭以节省 token）
- 🔄 单轮模式（seq512 限制，自动清空 history）
- ⚡ TPU 推理，约 5-8 tok/s

## API

```
POST /api/chat
Content-Type: application/json
Body: {"message": "你好"}

Response: SSE 流式
  data:"累积输出..."
  data:"__done__"
```

## 部署

见 [SKILL.md](./SKILL.md)。

## 文件

| 文件 | 说明 |
|------|------|
| `SKILL.md` | 完整部署文档（含 Python 3.10 ABI、seq512 限制说明） |
| `qwen3_web.py` | Flask Web 服务（含前端 HTML） |
