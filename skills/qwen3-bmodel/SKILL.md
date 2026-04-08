---
name: qwen3-bmodel
description: 在 SOPHON BM1684X 设备上部署 Qwen3-4B 对话模型服务，含 Flask Web 前端（SSE 流式输出、思考模式切换、单轮对话），通过 gssh 端口转发本地访问。
---

# Qwen3-bmodel Skill

在 SOPHON BM1684X 上部署 Qwen3-4B（AWQ W4BF16）Web 对话服务。

## 硬件 / 环境

- SOPHON BM1684X SOC (aarch64)
- bmodel：`/data/LLM-TPU/models/Qwen3/models/BM1684X/qwen3-4b-awq_w4bf16_seq512_bm1684x_1dev_*.bmodel`
- 推理扩展：`/data/LLM-TPU/models/Qwen3/python_demo/chat.cpython-310-aarch64-linux-gnu.so`（**cpython-310 编译，必须用 Python 3.10**）
- Python：`/data/AIGC-SDK/hub_venv/bin/python3.10`（系统 `/usr/bin/python3` 是 3.8，import `chat` 会失败）
- 依赖：`flask`、`transformers`（仅用 tokenizer，已在 hub_venv 中）

## 关键坑点

### 1. Python 版本必须 3.10
`chat.cpython-310-aarch64-linux-gnu.so` 是 cpython 3.10 ABI 编译产物。用 3.8 启动会报 `No module named 'chat'`（虽然 `.so` 文件就在 sys.path 里）。**固定用 `/data/AIGC-SDK/hub_venv/bin/python3.10`**。

### 2. bmodel 是 seq512，必须单轮对话
该 bmodel 编译时 `SEQLEN=512`，是 **prompt + 输出的总和**。`pipeline.Qwen2.stream_predict` 默认会累积 history，多轮对话几轮就会 `... (reach the maximal length)` 截断输出。

**解决方案**：Web 服务每次 `/api/chat` 结束后在 `finally` 中调用 `model.clear()` 重置 history，强制单轮模式。

```python
def generate():
    with model_lock:
        try:
            for answer, _ in model.stream_predict(message):
                ...
            yield 'data:"__done__"\n\n'
        except Exception as e:
            yield f'data:{json.dumps("__error__" + str(e))}\n\n'
        finally:
            try: model.clear()
            except Exception: pass
```

### 3. 前端 SSE 收到 `__done__` 必须跳出 while 外层并 cancel reader
服务端的 `finally: model.clear()` 会延迟 SSE 连接关闭，如果前端只 `break` 内层 for 循环继续 `reader.read()`，会导致 `isGenerating` 长时间卡住，**第二轮发送被阻塞**。

```js
let finished = false;
while(!finished){
  const {done, value} = await reader.read();
  if(done) break;
  ...
  for(const line of lines){
    ...
    if(tok === '__done__'){ finished = true; break; }
    ...
  }
}
try{ await reader.cancel(); }catch(e){}
```

### 4. 后台启动必须用 setsid，不能用 disown
Ubuntu 20.04 的 `/bin/sh` (dash) 没有 `disown`。用 `nohup ... &` 经 gssh exec 启动时，父 shell 退出会带走子进程。**用 `setsid` 脱离会话**：

```bash
gssh exec "cd /data/LLM-TPU/models/Qwen3/python_demo && \
  setsid /data/AIGC-SDK/hub_venv/bin/python3.10 qwen3_web.py \
  > /tmp/qwen3_web.log 2>&1 < /dev/null &"
```

## 部署步骤

### 1. 确认 bmodel 存在

```bash
gssh exec "ls /data/LLM-TPU/models/Qwen3/models/BM1684X/ \
           /data/LLM-TPU/models/Qwen3/python_demo/chat*.so"
```

若 bmodel 缺失，从 [LLM-TPU](https://github.com/sophgo/LLM-TPU) 仓库的 Qwen3 demo 下载对应 bmodel 到 `models/BM1684X/`。

### 2. 上传 Web 服务脚本

```bash
gssh scp -put qwen3_web.py /data/LLM-TPU/models/Qwen3/python_demo/qwen3_web.py
```

### 3. 启动服务

```bash
gssh exec "cd /data/LLM-TPU/models/Qwen3/python_demo && \
  setsid /data/AIGC-SDK/hub_venv/bin/python3.10 qwen3_web.py \
  > /tmp/qwen3_web.log 2>&1 < /dev/null &"
```

### 4. 等待模型加载（约 60 秒）

```bash
gssh exec "tail -10 /tmp/qwen3_web.log"
# 期待看到：
# Load Time: 57.xxx s
# [Qwen3] 模型加载完成
# * Running on http://0.0.0.0:5000
```

### 5. 建立本地端口转发

```bash
gssh forward -l 5000 -r 5000
```

### 6. 浏览器访问

打开 `http://localhost:5000`，即可：
- 与 Qwen3-4B 对话，SSE 流式输出
- 切换「思考模式」（默认关闭，追加 `/no_think` 以节省 token）
- 每条消息独立单轮（后端自动 clear history）

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | Web 前端 |
| GET  | `/api/status` | `{status: loading\|ready\|error, error}` |
| POST | `/api/chat` | body `{message}`，SSE 流式返回 |
| POST | `/api/clear` | 手动清空 history（单轮模式下无必要）|

### SSE 响应

```
data:"当前累积输出..."   # 逐步更新（累积而非增量）
data:"__done__"          # 结束
data:"__error__<msg>"    # 错误
```

## 停止服务

```bash
gssh exec "pkill -f qwen3_web.py"
```

## 性能参考（BM1684X, Qwen3-4B AWQ W4BF16 seq512）

| 阶段 | 耗时 |
|------|------|
| 模型加载 | ~57 秒 |
| 首 token（prefill） | ~1-2 秒 |
| 解码 | ~5-8 token/s |
| 最大上下文 | 512 tokens（prompt + 输出）|

## 如需更长上下文

seq512 是 bmodel 编译期硬编码，代码层面无法放宽。要获得更长输出需：
1. 用 TPU-MLIR 重新编译 seq2048/4096 的 bmodel，或
2. 下载 sophgo 官方提供的更大 seq 预编译 bmodel
