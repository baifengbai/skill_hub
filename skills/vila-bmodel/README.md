# VILA-bmodel — 视觉语言模型

基于 VILA-1.5-3B，在 BM1684X TPU 上运行视觉语言模型，支持图片和视频问答，提供 Web 界面和 SSE 流式 API。

## 功能

- 🖼️ 图片上传 + 文字提问
- 🎬 视频上传 + 文字提问（自动抽帧）
- 💬 SSE 流式输出，实时显示生成过程
- ⚡ TPU 推理，不依赖 transformers（纯 sentencepiece + numpy 实现）

## API

```
POST /api/infer
Content-Type: multipart/form-data
Body: media=<图片/视频> is_image=1|0 question=<问题>

Response: SSE 流式
  data:__vision__1.2s    # 视觉编码耗时
  data:__prefill__8.5s   # 预填充耗时
  data:Hello             # 逐 token 文本
  data:__end__           # 结束
```

## 部署

见 [SKILL.md](./SKILL.md)。

## 文件

| 文件 | 说明 |
|------|------|
| `SKILL.md` | 完整部署文档（含 transformers SIGILL 绕过方案） |
| `vila_web.py` | Flask Web 服务（含前端 HTML） |
