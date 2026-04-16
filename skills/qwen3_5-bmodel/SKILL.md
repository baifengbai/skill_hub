---
name: qwen3_5-bmodel
description: 在 SOPHON BM1684X 设备上部署 Qwen3.5-VL 多模态模型服务（图片/视频/文字问答），含 Flask Web 前端（SSE 流式输出、文件上传），通过 gssh 端口转发本地访问。
---

# Qwen3.5-bmodel Skill

在 SOPHON BM1684X 上部署 Qwen3.5-VL（2B int4 AutoRound W4BF16 seq2048）多模态 Web 对话服务，支持图片、视频、纯文字问答。

## 硬件 / 环境

- SOPHON BM1684X SOC (aarch64, Ubuntu 20.04)
- bmodel: `qwen3.5-2b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_*.bmodel`（~2.3G）
- 推理扩展: `chat.cpython-310-aarch64-linux-gnu.so`（需 Python 3.10 编译）
- Python: `/data/AIGC-SDK/hub_venv/bin/python3.10`
- 依赖: `flask`, `transformers>=5.5`, `torchvision`, `qwen_vl_utils`, `numpy`, `pillow`

## 关键坑点

### 1. transformers 版本必须 >= 5.x

Qwen3.5 的 `model_type: "qwen3_5"` 和 `Qwen3VLProcessor`/`Qwen3VLVideoProcessor` 在 transformers 4.x 中不存在。`AutoProcessor.from_pretrained()` 会回退为 `Qwen2TokenizerFast`（无 `.tokenizer` 属性），报 `has no attribute tokenizer`。

```bash
/data/AIGC-SDK/hub_venv/bin/pip3 install --upgrade transformers qwen_vl_utils
```

升级后 AutoProcessor 正确返回 `Qwen2_5_VLProcessor`（兼容 Qwen3.5）。

### 2. config 目录需要 processor_config.json

`AutoProcessor` 靠 `processor_config.json` 识别处理器类型。如果缺失，需手动创建：

```bash
echo '{"processor_class": "Qwen2_5_VLProcessor"}' > config/processor_config.json
```

`video_preprocessor_config.json` 中的 `video_processor_type` 也必须是 transformers 支持的类名。

### 3. chat.so 必须用 Python 3.10 编译

CMakeLists.txt 硬编码 `find_package(Python 3.10 REQUIRED)`。编译步骤：

```bash
cd python_demo
mkdir build && cd build
cmake .. \
  -DPython_EXECUTABLE=/data/AIGC-SDK/hub_venv/bin/python3.10 \
  -Dpybind11_DIR=/data/AIGC-SDK/hub_venv/lib/python3.10/site-packages/pybind11/share/cmake/pybind11
make -j4
cp chat.cpython-310-aarch64-linux-gnu.so ..
```

### 4. dfss 下载用 --enable_http

SFTP 模式下载新上传的文件可能报 `Unable to open file with SFTP`，加 `--enable_http` 切换 HTTP 协议可解决。

### 5. 前端 FormData + SSE 的错误处理

服务端返回 400/503（JSON 格式）时，前端不能直接用 `reader.read()` 当 SSE 解析，否则报"网络错误"。必须先检查 `resp.ok`：

```js
const resp = await fetch('/api/chat', {method:'POST', body:fd});
if (!resp.ok) {
  const err = await resp.json().catch(() => ({error: 'HTTP ' + resp.status}));
  // 显示错误
  throw {handled: true};
}
// 然后才读 SSE 流
```

### 6. 单轮模式（清空 history）

seq2048 相比 Qwen3 的 seq512 宽裕很多，但多轮累积仍会耗尽。`finally` 中调用 `model.model.clear_history()` + 重置 `model.history_max_posid = 0`。

### 7. setsid 启动

与 Qwen3 相同，dash 没有 `disown`，必须用 `setsid` 脱离会话。

## 部署步骤

### 1. 拉取代码

```bash
gssh exec "cd /data/LLM-TPU && git fetch origin refs/heads/main && git checkout origin/main -- models/Qwen3_5"
```

### 2. 升级依赖

```bash
gssh exec "/data/AIGC-SDK/hub_venv/bin/pip3 install --upgrade transformers qwen_vl_utils torchvision"
```

### 3. 编译 chat.so

```bash
gssh exec "cd /data/LLM-TPU/models/Qwen3_5/python_demo && mkdir -p build && cd build && cmake .. -DPython_EXECUTABLE=/data/AIGC-SDK/hub_venv/bin/python3.10 -Dpybind11_DIR=/data/AIGC-SDK/hub_venv/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 && make -j4 && cp chat.cpython-310-aarch64-linux-gnu.so .."
```

### 4. 下载 bmodel

```bash
gssh exec "mkdir -p /data/LLM-TPU/models/Qwen3_5/models/BM1684X && cd /data/LLM-TPU/models/Qwen3_5/models/BM1684X && /data/AIGC-SDK/hub_venv/bin/python3.10 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-2b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260415_111517.bmodel --enable_http"
```

约 2.3G，速度约 1.7 MB/s，需等待约 20 分钟。

### 5. 上传 Web 服务脚本

```bash
gssh scp -put qwen3_5_web.py /data/LLM-TPU/models/Qwen3_5/python_demo/qwen3_5_web.py
```

### 6. 启动服务

```bash
gssh exec "cd /data/LLM-TPU/models/Qwen3_5/python_demo && setsid /data/AIGC-SDK/hub_venv/bin/python3.10 qwen3_5_web.py > /tmp/qwen3_5_web.log 2>&1 < /dev/null &"
```

### 7. 等待模型加载（约 60-90 秒）

```bash
gssh exec "tail -10 /tmp/qwen3_5_web.log"
# 期待看到：
# [Qwen3.5] 模型加载完成
# * Running on http://0.0.0.0:5004
```

### 8. 建立本地端口转发

```bash
gssh forward -l 5004 -r 5004
```

### 9. 浏览器访问

打开 `http://localhost:5004`，可以：
- 上传图片/视频 + 文字提问（多模态问答）
- 纯文字对话
- SSE 流式输出，实时显示生成过程
- 显示 prefill 耗时、token 数、tps 等性能指标

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | Web 前端 |
| GET  | `/api/status` | `{status: loading|ready|error, error}` |
| POST | `/api/chat` | FormData: `message`（必填）+ `media`（可选文件），SSE 流式返回 |
| POST | `/api/clear` | 手动清空 history |

### SSE 响应格式

```
data:{"type":"perf","prefill":1.23}   # prefill 完成
data:"累积输出文本..."                 # 逐步更新（累积）
data:{"type":"perf","total":5.6,"tokens":42,"tps":8.5}  # 完成统计
data:"__done__"                        # 结束
data:"__error__<msg>"                  # 错误
```

### curl 示例

```bash
# 纯文字
curl -X POST http://localhost:5004/api/chat -F 'message=hello'

# 带图片
curl -X POST http://localhost:5004/api/chat -F 'message=描述这张图片' -F 'media=@photo.jpg'
```

## 停止服务

```bash
gssh exec "pkill -f qwen3_5_web.py"
```

## 性能参考（BM1684X, Qwen3.5-2B int4 W4BF16 seq2048）

| 阶段 | 耗时 |
|------|------|
| 模型加载 | ~60-90 秒 |
| 纯文字 prefill | ~0.1-0.2 秒 |
| 图片 prefill（含 VIT） | ~2-5 秒 |
| 解码 | ~5-10 tok/s |
| 最大上下文 | 2048 tokens（prompt + 输出）|
| 图片 token 计算 | 长 x 宽 / 32 / 32 |
| 视频 | 默认 1fps，尺寸为图片的 1/4 |
