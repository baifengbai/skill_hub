# VILA-bmodel Skill

在 SOPHON BM1684X 设备上部署 VILA-1.5-3B 视觉语言模型服务，支持图片和视频问答，含 Web 前端和 SSE 流式输出，通过 gssh 端口转发在本地浏览器访问。

> **重要**：不要安装 `transformers` 包（PyPI ARM wheel 在 aarch64 上触发 SIGILL）。本 skill 使用 `sentencepiece` + 纯 numpy/PIL 直接实现 tokenizer 和图像预处理。

## 硬件要求

- SOPHON BM1684X SOC (aarch64, Ubuntu 20.04)
- TPU 内存: 13.5 GB（LLM 1.6G + vision 887M，合计约 2.5G TPU）
- 系统内存: 建议开启 swap（防止 OOM）

## 模型路径（远程设备）

```
/data/sophon-demo/sample/Vila/models/BM1684X/vision_embedding_6batch.bmodel  (887M)
/data/sophon-demo/sample/Vila/models/BM1684X/llama_int4_seq2560.bmodel       (1.6G)
/data/sophon-demo/sample/Vila/python/config/image_processer/preprocessor_config.json
/data/sophon-demo/sample/Vila/python/config/llm_token/tokenizer.model
```

## Python 环境

使用系统 Python 3.8（`/usr/bin/python3`），需要以下包：
- `sophon.sail` — TPU 推理
- `cv2` (4.13.0) — 视频帧提取/图片读取
- `flask` (2.2.2) — Web 服务
- `numpy`, `Pillow` — 图像处理
- `sentencepiece 0.2.0` — tokenizer（**不用 transformers**）

## 部署步骤

### 1. 下载模型（在远程设备执行）

```bash
gssh exec "pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade -q"
gssh exec "mkdir -p /data/sophon-demo/sample/Vila/models/BM1684X"
gssh exec "nohup bash -c 'cd /data/sophon-demo/sample/Vila/models/BM1684X && python3 -m dfss --url=open@sophgo.com:sophon-demo/vila/vision_embedding_6batch.bmodel >> /data/vila_download.log 2>&1 && python3 -m dfss --url=open@sophgo.com:sophon-demo/vila/llama_int4_seq2560.bmodel >> /data/vila_download.log 2>&1 && echo DONE >> /data/vila_download.log' &"
# 等待下载完成（约 2.5G，需几分钟）
gssh exec "tail -5 /data/vila_download.log && ls -lh /data/sophon-demo/sample/Vila/models/BM1684X/"
```

### 2. 上传 Web 服务脚本

```bash
gssh scp -put vila_web.py /home/linaro/vila_web.py
```

### 3. 启动服务

```bash
gssh exec "nohup python3 /home/linaro/vila_web.py > /data/vila_web.log 2>&1 &"
```

### 4. 检查日志（模型加载约需 60-90 秒）

```bash
gssh exec "tail -10 /data/vila_web.log"
# 正常输出:
# [VILA] 模型加载完成
# Running on http://0.0.0.0:5003
```

### 5. 建立本地端口转发

```bash
gssh forward -l 5003 -r 5003
```

### 6. 浏览器访问

打开 `http://localhost:5003`，可以：
- 选择图片或视频 tab
- 上传图片/视频并输入问题
- 流式查看 VILA 的回答
- 查看视觉编码耗时、预填充耗时等性能统计

## API 说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | Web 前端页面 |
| GET  | `/api/status` | 返回模型加载状态 `{status, error}` |
| POST | `/api/infer` | 提交图片/视频问答，SSE 流式返回 |

### POST /api/infer 参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `media` | file | 图片或视频文件 |
| `is_image` | form field | `1` 表示图片，`0` 表示视频 |
| `question` | form field | 问题文本 |

### SSE 响应格式

```
data:__vision__1.2s      # 视觉编码完成，耗时
data:__prefill__8.5s     # 预填充完成，耗时
data:Hello               # 逐 token 文本
data:__end__             # 生成结束
data:__error__<msg>      # 发生错误
```

## 关键实现说明

### 不用 transformers 的原因

PyPI 上的 transformers ARM wheel 在 aarch64 (BM1684X) 上执行时触发 SIGILL（非法指令），exit code 132/-4。问题出在 `transformers.utils.import_utils` 和 `transformers.utils.versions`，疑似 Rust/C 扩展使用了该 ARM 核不支持的指令集。

### 替代实现

```python
# 图像预处理：直接读 preprocessor_config.json
class _SiglipImageProcessorLite:
    # 读取 size/mean/std，手动 resize+normalize+transpose

# Tokenizer：直接用 sentencepiece
class _SentencePieceLiteTokenizer:
    # sp.Load(tokenizer.model)
    # __call__: [bos_id] + sp.encode(text)
    # decode: sp.decode(ids)
```

### 图片输入 prompt 必须用 `<video>` 标记

无论输入是图片还是视频，媒体类型标记**始终用 `<video>`**，不能用 `<image>`：

```python
# 正确
prompt = f"... USER: {'<image>\n' * num_frames}<video>\\n {question}. ASSISTANT:"
# 错误（图片时用 <image> 会多出第 7 个 image token → 模型立即输出 EOS）
prompt = f"... USER: {'<image>\n' * num_frames}<image>\\n {question}. ASSISTANT:"
```

`<image>` 只作为帧占位符（每帧一个），`<video>` 是媒体类型标识符，两者用途不同。

### 中文乱码（多字节 UTF-8 字节回退 token）

LLaMA tokenizer 对非 ASCII 字符（中文、Emoji 等）使用字节回退 token（如 `中` = `<0xE4><0xB8><0xAD>`）。逐 token 调用 `sp.decode()` 会得到不完整 UTF-8 序列，输出 `□□□`。

**解决方案**：用 `sp.id_to_piece(token)` 检测字节 token，累积字节直到凑够完整 UTF-8 再 yield：

```python
byte_buf = b''
while token != 2:
    piece = sp.id_to_piece(token)
    if piece.startswith('<0x') and piece.endswith('>'):
        byte_buf += bytes([int(piece[3:-1], 16)])
        try:
            yield byte_buf.decode('utf-8'); byte_buf = b''
        except UnicodeDecodeError:
            pass  # 等待更多字节
    else:
        if byte_buf:
            yield byte_buf.decode('utf-8', errors='replace'); byte_buf = b''
        yield piece.replace('▁', ' ')
    token = forward_next()
```

### get_input_tensors 导致 heap corruption

在 BM1684X 上调用 `model.get_input_tensors(N_BLOCK_CACHE[i])` 会导致 `free(): invalid next size`，原因未知（疑似 sail Python binding bug）。

**解决方案（io_alone=0 模式）**：用 `sail.Tensor(handle, shape, dtype, False, True)` 创建独立张量，而不是复用模型内部张量：

```python
past_k = [sail.Tensor(handle, model.get_input_shape(N_BLOCK_CACHE[i], 3),
                      model.get_input_dtype(N_BLOCK_CACHE[i], 3), False, True)
          for i in range(NUM_LAYERS)]
```

## 性能（BM1684X, VILA-1.5-3B int4）

| 阶段 | 耗时 |
|------|------|
| 模型加载 | ~60-90 秒 |
| 视觉编码（6帧） | ~1-2 秒 |
| 预填充（~2560 tokens） | ~8-15 秒 |
| 逐 token 生成 | 约 1-2 token/s |

## 数据文件说明

| 路径 | 说明 |
|------|------|
| `/data/vila_uploads/` | 上传媒体临时存储（推理后自动删除） |
| `/data/vila_web.log` | 服务运行日志 |
| `/data/vila_download.log` | 模型下载日志 |

## 停止服务

```bash
gssh exec "pkill -f vila_web.py"
```
