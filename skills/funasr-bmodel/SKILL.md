---
name: funasr-bmodel
description: 在 SOPHON BM1684X 设备上部署 FunASR-bmodel 语音识别服务，含 Web 前端，通过 gssh 端口转发本地访问。
allowed-tools: Read, Write, Glob, Grep, Bash
version: 1.0.0
---

## 概述

在 SOPHON BM1684X（SOC 模式）设备上部署 [FunASR-bmodel](https://modelscope.cn/models/wlc952/FunASR-bmodel)，使用 TPU 加速推理，通过 Flask 提供 Web 服务，gssh 端口转发到本地浏览器访问。

**关键环境信息：**
- 设备：BM1684X SOC，TPU 内存 13.5GB，系统内存 1.5GB
- 系统：Ubuntu 20.04 aarch64
- 推理引擎：`tpu_perf.infer.SGInfer`（不依赖 sophon.sail）
- Python：需要 Python 3.10 虚拟环境（含 torch）
  - **不要用系统 Python 3.8**：PyPI 的 torch 2.x 在该 ARM CPU 上会报 `Illegal instruction`
  - 如设备已有 `/data/AIGC-SDK/hub_venv` 可直接使用；否则参考 README "通用前置条件" 创建 Python 3.10 虚拟环境

---

## 部署流程

### 1. 准备工作

**添加 swap（必须，防止加载模型时 OOM）：**
```bash
# /data 分区有足够空间（44GB），在上面创建 swap
fallocate -l 2G /data/swapfile
chmod 600 /data/swapfile
mkswap /data/swapfile
swapon /data/swapfile

# 开机自动挂载
echo '/data/swapfile none swap sw 0 0' >> /etc/fstab
```

**安装依赖：**
```bash
# ffmpeg 用于音频格式转换
apt-get install -y ffmpeg

# 在 Python 3.10 虚拟环境中安装依赖（以下用 $VENV 指代虚拟环境路径）
# 如有 AIGC-SDK：VENV=/data/AIGC-SDK/hub_venv；否则自行创建：VENV=/data/py310env
$VENV/bin/pip install librosa flask -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
$VENV/bin/pip install hydra-core kaldiio omegaconf six torch-complex tqdm -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装 tpu-perf（从项目自带 whl）
cd /data/FunASR-bmodel
$VENV/bin/pip install bmodel/tpu_perf-1.2.35-py3-none-manylinux2014_aarch64.whl
```

### 2. 克隆项目

```bash
cd /data
GIT_LFS_SKIP_SMUDGE=1 git clone https://modelscope.cn/models/wlc952/FunASR-bmodel.git
```

> git-lfs 不需要安装，加 `GIT_LFS_SKIP_SMUDGE=1` 跳过大文件，代码正常克隆。

### 3. 下载模型（用 dfss，机器上已预装）

```bash
# ASR 主模型（Paraformer Large）
cd /data/FunASR-bmodel/bmodel/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404/scripts
python3 -m dfss --url=open@sophgo.com:sophon-demo/FunASR/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404/BM1684X.zip
unzip BM1684X.zip -d ../models && rm BM1684X.zip

# VAD 模型
cd /data/FunASR-bmodel/bmodel/speech_fsmn_vad_zh-cn-16k-common/scripts
python3 -m dfss --url=open@sophgo.com:sophon-demo/FunASR/speech_fsmn_vad_zh-cn-16k-common/BM1684X.zip
unzip BM1684X.zip -d ../models && rm BM1684X.zip

# 标点模型
cd /data/FunASR-bmodel/bmodel/punc_ct-transformer_zh-cn-common-vocab272727/scripts
python3 -m dfss --url=open@sophgo.com:sophon-demo/FunASR/punc_ct-transformer_zh-cn-common-vocab272727/BM1684X.zip
unzip BM1684X.zip -d ../models && rm BM1684X.zip

# 说话人模型（可选，内存紧张可跳过）
cd /data/FunASR-bmodel/bmodel/speech_campplus_sv_zh-cn_16k-common/scripts
python3 -m dfss --url=open@sophgo.com:sophon-demo/FunASR/speech_campplus_sv_zh-cn_16k-common/BM1684X.zip
unzip BM1684X.zip -d ../models && rm BM1684X.zip
```

### 4. Web 服务（funasr_web.py）

保存为 `/data/FunASR-bmodel/funasr_web.py`：

```python
import os, sys, uuid, subprocess
sys.path.insert(0, '/data/FunASR-bmodel')
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
UPLOAD_DIR = '/tmp/funasr_uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)
model = None

def load_model():
    global model
    from funasr import AutoModel
    print("Loading FunASR models into TPU...")
    model = AutoModel(
        model="speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404/models/BM1684X",
        vad_model="speech_fsmn_vad_zh-cn-16k-common/models/BM1684X",
        punc_model="punc_ct-transformer_zh-cn-common-vocab272727/models/BM1684X",
        device="cpu",
        disable_update=True,
        disable_pbar=True,
        dev_id=0,
    )
    print("Models loaded!")

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>FunASR 语音识别</title>
<!-- ... 完整 HTML 见部署后的 /data/FunASR-bmodel/funasr_web.py ... -->
</head>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/recognize', methods=['POST'])
def recognize():
    if model is None:
        return jsonify({'error': '模型未加载'}), 503
    f = request.files.get('audio')
    if not f:
        return jsonify({'error': '未收到音频文件'}), 400
    ext = os.path.splitext(f.filename)[1] or '.webm'
    raw_path = os.path.join(UPLOAD_DIR, f'{uuid.uuid4()}{ext}')
    wav_path = raw_path.replace(ext, '.wav')
    try:
        f.save(raw_path)
        subprocess.run(['ffmpeg', '-y', '-i', raw_path,
            '-ar', '16000', '-ac', '1', '-f', 'wav', wav_path],
            capture_output=True, check=True)
        import time; t0 = time.time()
        res = model.generate(input=wav_path, batch_size_s=300)
        elapsed = round(time.time() - t0, 2)
        text = res[0].get('text', '') if res else ''
        return jsonify({'text': text, 'inference_time': elapsed})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        for p in [raw_path, wav_path]:
            try: os.remove(p)
            except: pass

if __name__ == '__main__':
    os.chdir('/data/FunASR-bmodel/bmodel')
    load_model()
    app.run(host='0.0.0.0', port=5001, threaded=False)
```

### 5. 启动服务

```bash
# 后台启动，日志写文件
# $VENV 为 Python 3.10 虚拟环境路径
nohup $VENV/bin/python3 /data/FunASR-bmodel/funasr_web.py > /tmp/funasr_web.log 2>&1 &

# 观察启动日志（模型加载约需 60~90 秒）
tail -f /tmp/funasr_web.log
# 看到 "Models loaded!" 和 "Running on http://0.0.0.0:5001" 表示成功
```

### 6. 本地访问（gssh 端口转发）

```bash
gssh forward -l 5001 -r 5001
# 打开浏览器访问 http://localhost:5001
```

---

## 常见问题

### OOM 导致设备重启
- **现象**：运行推理时设备崩溃重启
- **原因**：模型文件加载时会先进系统内存再传 TPU，峰值占用超过 1.5GB
- **解决**：务必先创建 swap（见步骤 1），有 swap 后不再崩溃

### Illegal instruction（torch）
- **现象**：`python3 -c 'import torch'` 报 `Illegal instruction`
- **原因**：PyPI 的 torch 2.x aarch64 wheel 使用了该 ARM CPU 不支持的指令集
- **解决**：使用 Python 3.10 虚拟环境中编译好的 torch（如 AIGC-SDK 的 hub_venv 或自建的 py310env）

### 模型加载很慢
- 正常现象，encoder_fp16 315MB + 其他模型合计约 400MB 需传入 TPU
- 首次启动约需 60~90 秒，之后请求响应正常

### 查看 TPU 内存使用
```bash
/opt/sophon/libsophon-0.5.1/bin/bm-smi --noloop --file=/tmp/s.txt && grep 'Memory-Usage' /tmp/s.txt
# 输出示例：0MB/13816MB → 模型加载后约占 800MB/13816MB
```

---

## 关键路径速查

| 内容 | 路径 |
|------|------|
| 项目根目录 | `/data/FunASR-bmodel/` |
| bmodel 文件 | `/data/FunASR-bmodel/bmodel/<model>/models/BM1684X/` |
| Web 服务 | `/data/FunASR-bmodel/funasr_web.py` |
| 服务日志 | `/tmp/funasr_web.log` |
| Python 3.10 环境 | `$VENV/bin/python3`（AIGC-SDK 的 hub_venv 或自建 py310env）|
| bm-smi | `/opt/sophon/libsophon-0.5.1/bin/bm-smi` |
| swap 文件 | `/data/swapfile`（2GB） |
