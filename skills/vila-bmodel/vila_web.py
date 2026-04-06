import os
import sys
import time
import uuid
import threading
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context, send_file
from PIL import Image

# ── 路径 ──────────────────────────────────────────────────────────────────────
VILA_BASE    = '/data/sophon-demo/sample/Vila'
LLM_BMODEL   = f'{VILA_BASE}/models/BM1684X/llama_int4_seq2560.bmodel'
VISION_BMODEL= f'{VILA_BASE}/models/BM1684X/vision_embedding_6batch.bmodel'
CONFIG_DIR   = f'{VILA_BASE}/python/config'
UPLOAD_DIR   = '/data/vila_uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

sys.path.insert(0, f'{VILA_BASE}/python')
os.chdir(VILA_BASE)

import sophon.sail as sail
import sentencepiece as spm

# ── 最小化替代 transformers（避免 aarch64 SIGILL）────────────────────────────

class _SiglipImageProcessorLite:
    """Minimal SigLIP preprocessor: reads preprocessor_config.json, resize + normalize."""
    def __init__(self, config_dir):
        cfg_path = os.path.join(config_dir, 'preprocessor_config.json')
        with open(cfg_path) as f:
            cfg = json.load(f)
        size_cfg = cfg.get('size', {})
        if isinstance(size_cfg, int):
            h = w = size_cfg
        else:
            h = size_cfg.get('height', size_cfg.get('shortest_edge', 384))
            w = size_cfg.get('width',  size_cfg.get('shortest_edge', 384))
        self.size = {'height': h, 'width': w}
        self.mean = np.array(cfg.get('image_mean', [0.5, 0.5, 0.5]), dtype=np.float32)
        self.std  = np.array(cfg.get('image_std',  [0.5, 0.5, 0.5]), dtype=np.float32)

    def preprocess(self, img, return_tensors='np'):
        arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)          # HWC -> CHW
        return {'pixel_values': arr[np.newaxis]}  # (1, C, H, W)


class _TokenizerResult:
    def __init__(self, ids):
        self.input_ids = ids


class _SentencePieceLiteTokenizer:
    """Minimal LLaMA tokenizer wrapper using sentencepiece directly."""
    def __init__(self, config_dir):
        model_path = os.path.join(config_dir, 'tokenizer.model')
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.bos_token_id = self.sp.bos_id()

    def __call__(self, text):
        # add_bos=True matches HuggingFace LLaMA tokenizer default behavior
        ids = [self.bos_token_id] + self.sp.encode(text, out_type=int)
        return _TokenizerResult(ids)

    def decode(self, ids):
        return self.sp.decode(ids)

app = Flask(__name__)

# ── 工具函数（来自 vila.py）──────────────────────────────────────────────────
def opencv_extract_frames(video_file, num_frames):
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    images, count = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count in frame_indices:
            try:
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            except Exception:
                pass
            if len(images) >= num_frames:
                break
        count += 1
    cap.release()
    return images, len(images)

def load_image_as_rgb(image_path, num_frames):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f'无法读取图片: {image_path}')
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return [pil] * num_frames, num_frames

def process_images(images, image_processor):
    out = []
    for img in images:
        img = img.convert('RGB')
        sz  = image_processor.size
        img = img.resize((sz['height'], sz['width']))
        arr = image_processor.preprocess(img, return_tensors='np')['pixel_values'][0]
        out.append(arr)
    return np.stack(out)

def tokenizer_image_token(prompt, tokenizer, image_token_index=-200):
    chunks = [tokenizer(c).input_ids for c in prompt.split('<image>')]
    def insert_sep(X, sep):
        return [e for sub in zip(X, [sep]*len(X)) for e in sub][:-1]
    ids, offset = [], 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        ids.append(chunks[0][0])
    for cid, x in enumerate(insert_sep(chunks, [image_token_index]*(offset+1))):
        ids.extend(x if (cid == 0 and not offset) else x[offset:])
    return ids

# ── 模型引擎 ──────────────────────────────────────────────────────────────────
class VilaEngine:
    def __init__(self):
        target = sail.Handle(0).get_target()
        if target in ('BM1688', 'CV186AH'):
            flag = sail.BmrtFlag.BM_RUNTIME_SHARE_MEM
            self.model        = sail.EngineLLM(LLM_BMODEL,    flag, [0])
            self.vision_model = sail.EngineLLM(VISION_BMODEL, flag, [0])
        else:
            self.model        = sail.EngineLLM(LLM_BMODEL,    [0])
            self.vision_model = sail.EngineLLM(VISION_BMODEL, [0])

        self.handle          = sail.Handle(0)
        self.image_processor = _SiglipImageProcessorLite(f'{CONFIG_DIR}/image_processer')
        self.tokenizer       = _SentencePieceLiteTokenizer(f'{CONFIG_DIR}/llm_token')

        self.graph_names   = self.model.get_graph_names()
        self.NUM_LAYERS    = (len(self.graph_names) - 3) // 2
        self.N_VE          = 'vision_embedding'
        self.N_EMBED       = 'embedding'
        self.N_EMBED_CACHE = 'embedding_cache'
        self.N_LM          = 'lm_head'
        self.N_BLOCK       = [f'block_{i}' for i in range(self.NUM_LAYERS)]
        self.N_BLOCK_CACHE = [f'block_cache_{i}' for i in range(self.NUM_LAYERS)]

        self._it, self._ot = {}, {}
        self._setup_tensors()

        _, self.SEQ_LEN, self.HIDDEN_SIZE   = self.model.get_input_shape(self.N_BLOCK[0], 0)
        _, _, self.NUM_HEADS, self.HEAD_DIM = self.model.get_output_shape(self.N_BLOCK[0], 1)
        self.lock        = threading.Lock()
        self.token_length = 0

    def _dtype(self, d):
        return {sail.Dtype.BM_FLOAT32: np.float32, sail.Dtype.BM_FLOAT16: np.float16,
                sail.Dtype.BM_INT32: np.int32, sail.Dtype.BM_BFLOAT16: np.uint16}.get(d, np.float16)

    def _mk_in(self, net, idx):
        return sail.Tensor(self.handle, self.model.get_input_shape(net, idx),
                           self.model.get_input_dtype(net, idx), False, True)

    def _mk_out(self, net, idx):
        return sail.Tensor(self.handle, self.model.get_output_shape(net, idx),
                           self.model.get_output_dtype(net, idx), False, True)

    def _setup_tensors(self):
        it, ot = self._it, self._ot
        it[self.N_VE] = self.vision_model.create_max_input_tensors(self.N_VE)
        ot[self.N_VE] = self.vision_model.create_max_output_tensors(self.N_VE)
        self.num_frames, self.vision_token_len, _ = ot[self.N_VE][0].shape()

        block0_in = self.model.create_max_input_tensors(self.N_BLOCK[0])
        hs = block0_in[0]; pid = block0_in[1]; amask = block0_in[2]
        # io_alone=0: create past_k/v as fresh tensors (get_input_tensors causes heap corruption on BM1684X)
        past_k = [self._mk_in(self.N_BLOCK_CACHE[i], 3) for i in range(self.NUM_LAYERS)]
        past_v = [self._mk_in(self.N_BLOCK_CACHE[i], 4) for i in range(self.NUM_LAYERS)]

        it[self.N_EMBED] = self.model.create_max_input_tensors(self.N_EMBED)
        ot[self.N_EMBED] = self.model.create_max_output_tensors(self.N_EMBED)

        for i in range(self.NUM_LAYERS):
            it[self.N_BLOCK[i]] = {0: hs, 1: pid, 2: amask}
            ot[self.N_BLOCK[i]] = {0: hs, 1: past_k[i], 2: past_v[i]}

        it[self.N_LM] = self.model.create_max_input_tensors(self.N_LM)
        ot[self.N_LM] = self.model.create_max_output_tensors(self.N_LM)
        it[self.N_EMBED_CACHE] = ot[self.N_LM]
        ot[self.N_EMBED_CACHE] = self.model.create_max_output_tensors(self.N_EMBED_CACHE)

        pid_next  = self._mk_in(self.N_BLOCK_CACHE[0], 1)
        amask_next= self._mk_in(self.N_BLOCK_CACHE[0], 2)
        pk_cache  = [self._mk_out(self.N_BLOCK_CACHE[i], 1) for i in range(self.NUM_LAYERS)]
        pv_cache  = [self._mk_out(self.N_BLOCK_CACHE[i], 2) for i in range(self.NUM_LAYERS)]

        for i in range(self.NUM_LAYERS):
            it[self.N_BLOCK_CACHE[i]] = {0: ot[self.N_EMBED_CACHE][0],
                                          1: pid_next, 2: amask_next,
                                          3: past_k[i], 4: past_v[i]}
            ot[self.N_BLOCK_CACHE[i]] = {0: ot[self.N_EMBED_CACHE][0],
                                          1: pk_cache[i], 2: pv_cache[i]}

    def _encode_media(self, media_path, is_image):
        if is_image:
            imgs, _ = load_image_as_rgb(media_path, self.num_frames)
        else:
            imgs, _ = opencv_extract_frames(media_path, self.num_frames)
        arr = process_images(imgs, self.image_processor).astype(np.float16)
        self._it[self.N_VE][0].update_data(arr.view(np.uint16))
        self.vision_model.process(self.N_VE, self._it[self.N_VE], self._ot[self.N_VE])

    def _forward_first(self, tokens):
        SL, HS = self.SEQ_LEN, self.HIDDEN_SIZE
        tok_arr = np.array(tokens)
        img_idx = np.where(tok_arr == -200)[0]
        img_idx = np.append(np.insert(img_idx, 0, -1), len(tokens))
        tok_arr[tok_arr == -200] = 0

        self.token_length = len(tokens) + (self.vision_token_len - 1) * self.num_frames

        ids = np.zeros(SL, self._dtype(self._it[self.N_EMBED][0].dtype()))
        ids[:len(tokens)] = tok_arr
        self._it[self.N_EMBED][0].update_data(ids.reshape(self._it[self.N_EMBED][0].shape()))
        self.model.process(self.N_EMBED, self._it[self.N_EMBED], self._ot[self.N_EMBED])

        off = 0
        for i in range(len(img_idx) - 1):
            seg = (img_idx[i+1] - img_idx[i] - 1) * HS
            self._it[self.N_BLOCK[0]][0].sync_d2d(
                self._ot[self.N_EMBED][0], (img_idx[i]+1)*HS, off, seg)
            off += seg
            if i < self.num_frames:
                vseg = self.vision_token_len * HS
                self._it[self.N_BLOCK[0]][0].sync_d2d(
                    self._ot[self.N_VE][0], i*vseg, off, vseg)
                off += vseg

        tl = self.token_length
        pid = np.zeros(SL, self._dtype(self._it[self.N_BLOCK[0]][1].dtype()))
        pid[:tl] = np.arange(tl)
        amask = np.full((SL, SL), -10000.0,
                        dtype=self._dtype(self._it[self.N_BLOCK[0]][2].dtype()))
        for i in range(tl): amask[i, :i+1] = 0

        self._it[self.N_BLOCK[0]][1].update_data(pid.reshape(self._it[self.N_BLOCK[0]][1].shape()))
        self._it[self.N_BLOCK[0]][2].update_data(
            amask.reshape(self._it[self.N_BLOCK[0]][2].shape()).view(np.uint16))

        for i in range(self.NUM_LAYERS):
            self.model.process(self.N_BLOCK[i], self._it[self.N_BLOCK[i]], self._ot[self.N_BLOCK[i]])

        self._it[self.N_LM][0].sync_d2d(
            self._ot[self.N_BLOCK[self.NUM_LAYERS-1]][0], (tl-1)*HS, 0, HS)
        self.model.process(self.N_LM, self._it[self.N_LM], self._ot[self.N_LM])
        return int(self._ot[self.N_LM][0].asnumpy())

    def _forward_next(self):
        self.token_length += 1
        tl = self.token_length
        SL  = self.SEQ_LEN
        pid = np.array(tl - 1, self._dtype(self._it[self.N_BLOCK_CACHE[0]][1].dtype()))
        amask = np.zeros(SL+1, self._dtype(self._it[self.N_BLOCK_CACHE[0]][2].dtype()))
        for i in range(tl-1, SL): amask[i] = -10000.0

        self.model.process(self.N_EMBED_CACHE, self._it[self.N_EMBED_CACHE], self._ot[self.N_EMBED_CACHE])
        self._it[self.N_BLOCK_CACHE[0]][1].update_data(
            pid.reshape(self._it[self.N_BLOCK_CACHE[0]][1].shape()))
        self._it[self.N_BLOCK_CACHE[0]][2].update_data(
            amask.reshape(self._it[self.N_BLOCK_CACHE[0]][2].shape()).view(np.uint16))

        for i in range(self.NUM_LAYERS):
            self.model.process(self.N_BLOCK_CACHE[i],
                               self._it[self.N_BLOCK_CACHE[i]],
                               self._ot[self.N_BLOCK_CACHE[i]])
            self._it[self.N_BLOCK_CACHE[i]][3].sync_d2d(
                self._ot[self.N_BLOCK_CACHE[i]][1], 0,
                (tl-1)*self.NUM_HEADS*self.HEAD_DIM,
                self.NUM_HEADS*self.HEAD_DIM)
            self._it[self.N_BLOCK_CACHE[i]][4].sync_d2d(
                self._ot[self.N_BLOCK_CACHE[i]][2], 0,
                (tl-1)*self.NUM_HEADS*self.HEAD_DIM,
                self.NUM_HEADS*self.HEAD_DIM)

        self._it[self.N_LM][0] = self._ot[self.N_BLOCK_CACHE[self.NUM_LAYERS-1]][0]
        self.model.process(self.N_LM, self._it[self.N_LM], self._ot[self.N_LM])
        return int(self._ot[self.N_LM][0].asnumpy())

    def infer_stream(self, media_path, is_image, question):
        """生成器：逐 token 输出，前缀特殊事件用 __xxx__ 包裹"""
        with self.lock:
            t0 = time.time()
            self._encode_media(media_path, is_image)
            yield f'__vision__{time.time()-t0:.1f}s'

            imgs_prompt = '<image>\n' * self.num_frames
            # 参考官方 vila.py：无论图片还是视频，媒体类型标记始终用 <video>
            # 用 <image> 会多出一个 -200 token (7 vs 6)，导致模型立即输出 EOS
            prompt = (f'A chat between a curious user and an artificial intelligence assistant. '
                      f'The assistant gives helpful, detailed, and polite answers to the user\'s questions. '
                      f'USER: {imgs_prompt}<video>\\n {question}. ASSISTANT:')
            tokens = tokenizer_image_token(prompt, self.tokenizer)

            t1 = time.time()
            token = self._forward_first(tokens)
            yield f'__prefill__{time.time()-t1:.1f}s'

            # 逐 token 流式解码，正确处理多字节 UTF-8（中文/Emoji 等）
            # 部分 token 是字节回退 token（如 <0xE4><0xB8><0xAD> = 中），
            # 需要累积字节直到构成完整 UTF-8 字符再 yield
            byte_buf = b''
            while token != 2:
                piece = self.tokenizer.sp.id_to_piece(token)
                if piece.startswith('<0x') and piece.endswith('>'):
                    # 字节回退 token，累积字节
                    byte_buf += bytes([int(piece[3:-1], 16)])
                    try:
                        text = byte_buf.decode('utf-8')
                        yield text
                        byte_buf = b''
                    except UnicodeDecodeError:
                        pass  # 等待更多字节
                else:
                    if byte_buf:
                        yield byte_buf.decode('utf-8', errors='replace')
                        byte_buf = b''
                    yield piece.replace('\u2581', ' ')  # ▁ → 空格
                token = self._forward_next()
            if byte_buf:
                yield byte_buf.decode('utf-8', errors='replace')
            yield '__end__'


# ── 全局模型实例（异步加载）───────────────────────────────────────────────────
engine       = None
engine_status = 'loading'   # loading | ready | error
engine_error  = ''

def _load_engine():
    global engine, engine_status, engine_error
    try:
        print('[VILA] 加载模型中...', flush=True)
        engine = VilaEngine()
        engine_status = 'ready'
        print('[VILA] 模型加载完成', flush=True)
    except Exception as e:
        import traceback
        engine_error  = str(e) + '\n' + traceback.format_exc()
        engine_status = 'error'
        print(f'[VILA] 模型加载失败: {e}', flush=True)

threading.Thread(target=_load_engine, daemon=True).start()

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VILA — 视觉语言模型</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#0d1117;color:#e6edf3;min-height:100vh;display:flex;flex-direction:column}
.header{background:rgba(255,255,255,.03);border-bottom:1px solid rgba(255,255,255,.07);
  padding:16px 28px;display:flex;align-items:center;gap:14px}
.header h1{font-size:18px;font-weight:700}
.badge{padding:3px 10px;border-radius:10px;font-size:11px;border:1px solid;margin-left:6px}
.badge.ready{background:rgba(46,160,67,.15);color:#3fb950;border-color:rgba(46,160,67,.3)}
.badge.loading{background:rgba(255,165,0,.15);color:#ffa500;border-color:rgba(255,165,0,.3)}
.badge.error{background:rgba(239,68,68,.15);color:#fca5a5;border-color:rgba(239,68,68,.3)}
.main{display:flex;flex:1;overflow:hidden}
/* 左侧输入栏 */
.sidebar{width:320px;border-right:1px solid rgba(255,255,255,.07);
  padding:20px;display:flex;flex-direction:column;gap:14px;overflow-y:auto}
.label{font-size:12px;color:#8b949e;margin-bottom:6px;font-weight:500}
.upload-zone{border:2px dashed rgba(88,166,255,.25);border-radius:12px;padding:24px;
  text-align:center;cursor:pointer;transition:.2s;color:#8b949e;font-size:13px;
  position:relative;min-height:120px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:6px}
.upload-zone:hover,.upload-zone.drag{border-color:#58a6ff;background:rgba(88,166,255,.05)}
.upload-zone input{display:none}
.upload-zone img,.upload-zone video{max-width:100%;max-height:150px;border-radius:8px;margin-top:8px}
.media-icon{font-size:28px}
.tab-row{display:flex;gap:6px;margin-bottom:2px}
.tab{flex:1;padding:7px;border-radius:8px;border:1px solid rgba(255,255,255,.1);
  background:none;color:#8b949e;font-size:12px;cursor:pointer;transition:.2s}
.tab.active{background:rgba(88,166,255,.15);color:#58a6ff;border-color:rgba(88,166,255,.3)}
textarea{width:100%;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);
  border-radius:10px;padding:10px 12px;color:#e6edf3;font-size:13px;resize:vertical;
  min-height:80px;font-family:inherit}
textarea:focus{outline:none;border-color:rgba(88,166,255,.4)}
.send-btn{width:100%;padding:11px;background:linear-gradient(135deg,#238636,#2ea043);
  border:none;border-radius:10px;color:#fff;font-size:14px;font-weight:600;
  cursor:pointer;transition:.2s}
.send-btn:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(46,160,67,.3)}
.send-btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
/* 右侧对话区 */
.chat-area{flex:1;display:flex;flex-direction:column;overflow:hidden}
.chat-scroll{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:820px;width:100%}
.msg.user{align-self:flex-end}
.msg.assistant{align-self:flex-start}
.msg-header{font-size:11px;color:#8b949e;margin-bottom:5px;display:flex;align-items:center;gap:6px}
.msg-body{padding:12px 16px;border-radius:12px;font-size:14px;line-height:1.65;white-space:pre-wrap;word-break:break-word}
.msg.user .msg-body{background:rgba(88,166,255,.12);border:1px solid rgba(88,166,255,.2)}
.msg.assistant .msg-body{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08)}
.msg-thumb{height:60px;width:auto;border-radius:6px;margin-bottom:6px;display:block}
.perf{font-size:11px;color:#484f58;margin-top:5px}
.typing{display:inline-block;width:8px;height:14px;background:#58a6ff;
  border-radius:2px;animation:blink .7s infinite;vertical-align:middle;margin-left:2px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.empty-hint{text-align:center;color:#484f58;padding:60px 20px;font-size:14px}
/* 模型加载遮罩 */
.overlay{position:fixed;inset:0;background:rgba(13,17,23,.9);display:flex;
  flex-direction:column;align-items:center;justify-content:center;gap:16px;z-index:100}
.overlay.hide{display:none}
.spinner{width:44px;height:44px;border:4px solid rgba(255,255,255,.1);
  border-top-color:#58a6ff;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.overlay p{color:#8b949e;font-size:14px}
</style>
</head>
<body>

<div class="overlay" id="overlay">
  <div class="spinner"></div>
  <p id="overlayMsg">模型加载中，请稍候...</p>
</div>

<div class="header">
  <h1>VILA 视觉语言模型<span class="badge loading" id="statusBadge">加载中</span></h1>
  <span style="color:#484f58;font-size:12px;margin-left:auto">VILA-1.5-3B · BM1684X int4 · TPU 加速</span>
  <button onclick="clearChat()" style="margin-left:16px;padding:5px 14px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:8px;color:#8b949e;font-size:12px;cursor:pointer" onmouseover="this.style.color='#e6edf3'" onmouseout="this.style.color='#8b949e'">清空对话</button>
</div>

<div class="main">
  <!-- 左栏 -->
  <div class="sidebar">
    <div>
      <div class="label">输入类型</div>
      <div class="tab-row">
        <button class="tab active" id="tabImg" onclick="switchTab('image')">🖼 图片</button>
        <button class="tab" id="tabVid" onclick="switchTab('video')">🎬 视频</button>
      </div>
    </div>

    <div>
      <div class="label">上传媒体</div>
      <div class="upload-zone" id="zone" onclick="document.getElementById('mediaInput').click()">
        <div class="media-icon">📂</div>
        <div>点击或拖入文件</div>
        <small id="zoneHint">支持 JPG / PNG / BMP</small>
        <input type="file" id="mediaInput" accept="image/*" onchange="onMedia(this)">
      </div>
      <div id="mediaPreview" style="margin-top:8px"></div>
    </div>

    <div>
      <div class="label">问题</div>
      <textarea id="question" rows="3" placeholder="请描述这张图片..." onkeydown="onKey(event)"></textarea>
    </div>

    <button class="send-btn" id="sendBtn" onclick="sendQuery()" disabled>发送</button>

    <div style="font-size:11px;color:#484f58;line-height:1.6">
      提示：模型首次推理需加载 KV 缓存，约需 30-60 秒预热；后续响应更快。
    </div>
  </div>

  <!-- 右侧对话区 -->
  <div class="chat-area">
    <div class="chat-scroll" id="chatScroll">
      <div class="empty-hint" id="emptyHint">
        <div style="font-size:40px;margin-bottom:12px">🤖</div>
        上传图片或视频，提问即可开始对话
      </div>
    </div>
  </div>
</div>

<script>
let mediaFile = null, mediaTab = 'image', isReady = false;

// ── 状态轮询 ─────────────────────────────────────────────────────────────────
(async function poll(){
  const res = await fetch('/api/status');
  const d   = await res.json();
  const badge = document.getElementById('statusBadge');
  badge.className = 'badge ' + d.status;
  badge.textContent = {loading:'加载中', ready:'就绪', error:'加载失败'}[d.status] || d.status;
  if(d.status === 'ready'){
    isReady = true;
    document.getElementById('overlay').classList.add('hide');
    document.getElementById('sendBtn').disabled = !mediaFile;
  } else if(d.status === 'error'){
    document.getElementById('overlayMsg').textContent = '模型加载失败: ' + d.error;
    document.getElementById('overlay').querySelector('.spinner').style.display = 'none';
  } else {
    setTimeout(poll, 3000);
  }
})();

// ── 媒体类型切换 ──────────────────────────────────────────────────────────────
function switchTab(t){
  mediaTab = t;
  const inp = document.getElementById('mediaInput');
  inp.accept = t === 'image' ? 'image/*' : 'video/*';
  document.getElementById('tabImg').classList.toggle('active', t==='image');
  document.getElementById('tabVid').classList.toggle('active', t==='video');
  document.getElementById('zoneHint').textContent = t === 'image'
    ? '支持 JPG / PNG / BMP' : '支持 MP4 / AVI / MOV';
  document.getElementById('question').placeholder = t === 'image'
    ? '请描述这张图片...' : '这段视频里发生了什么？';
  mediaFile = null;
  document.getElementById('mediaPreview').innerHTML = '';
  document.getElementById('sendBtn').disabled = true;
}

// ── 媒体选择 ──────────────────────────────────────────────────────────────────
const zone = document.getElementById('zone');
zone.ondragover = e=>{ e.preventDefault(); zone.classList.add('drag'); };
zone.ondragleave = ()=>zone.classList.remove('drag');
zone.ondrop = e=>{ e.preventDefault(); zone.classList.remove('drag');
  if(e.dataTransfer.files[0]) processMedia(e.dataTransfer.files[0]); };

function onMedia(inp){ if(inp.files[0]) processMedia(inp.files[0]); }
function processMedia(f){
  mediaFile = f;
  const url = URL.createObjectURL(f);
  const prev = document.getElementById('mediaPreview');
  prev.innerHTML = mediaTab === 'image'
    ? `<img src="${url}" style="max-width:100%;max-height:150px;border-radius:8px">`
    : `<video src="${url}" style="max-width:100%;max-height:120px;border-radius:8px" controls muted></video>`;
  if(isReady) document.getElementById('sendBtn').disabled = false;
}

// ── 发送 ─────────────────────────────────────────────────────────────────────
function onKey(e){ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendQuery(); } }

async function sendQuery(){
  if(!mediaFile || !isReady) return;
  const q = document.getElementById('question').value.trim();
  if(!q){ alert('请输入问题'); return; }

  document.getElementById('sendBtn').disabled = true;
  document.getElementById('emptyHint')?.remove();

  // 用户消息
  const thumbUrl = URL.createObjectURL(mediaFile);
  const thumb = mediaTab === 'image'
    ? `<img class="msg-thumb" src="${thumbUrl}">`
    : `<video class="msg-thumb" src="${thumbUrl}" muted></video>`;
  addMsg('user', q, thumb);

  // 助手消息占位
  const aid = 'msg-' + Date.now();
  const bodyEl = addMsg('assistant', '', '', aid);
  bodyEl.innerHTML = '<span class="typing"></span>';

  // 上传 + 流式请求
  const fd = new FormData();
  fd.append('media', mediaFile);
  fd.append('is_image', mediaTab === 'image' ? '1' : '0');
  fd.append('question', q);

  let perfInfo = '', fullText = '';
  try{
    const resp = await fetch('/api/infer', {method:'POST', body:fd});
    const reader = resp.body.getReader();
    const dec    = new TextDecoder();
    let buf = '';

    while(true){
      const {done, value} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for(const line of lines){
        if(!line.startsWith('data:')) continue;
        const tok = line.slice(5);
        if(tok.startsWith('__vision__')){
          perfInfo += `视觉编码: ${tok.slice(10)}  `;
        } else if(tok.startsWith('__prefill__')){
          perfInfo += `预填充: ${tok.slice(11)}  `;
        } else if(tok === '__end__'){
          // done
        } else if(tok.startsWith('__error__')){
          fullText += '\n[错误] ' + tok.slice(9);
        } else {
          fullText += tok;
          bodyEl.innerHTML = escHtml(fullText) + '<span class="typing"></span>';
          document.getElementById('chatScroll').scrollTop = 999999;
        }
      }
    }
  }catch(e){
    fullText += '\n[网络错误] ' + e.message;
  }

  bodyEl.innerHTML = escHtml(fullText);
  if(perfInfo){
    const perf = document.createElement('div');
    perf.className = 'perf'; perf.textContent = perfInfo;
    bodyEl.parentNode.appendChild(perf);
  }
  document.getElementById('chatScroll').scrollTop = 999999;
  document.getElementById('sendBtn').disabled = false;
}

function addMsg(role, text, extraHtml='', id=''){
  const scroll = document.getElementById('chatScroll');
  const msg = document.createElement('div');
  msg.className = 'msg ' + role;
  if(id) msg.id = id;
  const who = role === 'user' ? '👤 用户' : '🤖 VILA';
  msg.innerHTML = `
    <div class="msg-header">${who}</div>
    ${extraHtml}
    <div class="msg-body">${escHtml(text)}</div>`;
  scroll.appendChild(msg);
  scroll.scrollTop = 999999;
  return msg.querySelector('.msg-body');
}

function escHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function clearChat(){
  const scroll = document.getElementById('chatScroll');
  scroll.innerHTML = '<div class="empty-hint" id="emptyHint"><div style="font-size:40px;margin-bottom:12px">🤖</div>上传图片或视频，提问即可开始对话</div>';
}
</script>
</body>
</html>"""

# ── Flask 路由 ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/status')
def api_status():
    return jsonify({'status': engine_status,
                    'error': engine_error[:200] if engine_error else ''})

@app.route('/api/infer', methods=['POST'])
def api_infer():
    if engine_status != 'ready':
        return jsonify({'error': '模型未就绪: ' + engine_status}), 503

    f        = request.files.get('media')
    is_image = request.form.get('is_image', '1') == '1'
    question = request.form.get('question', '').strip()

    if not f or not question:
        return jsonify({'error': '缺少 media 或 question'}), 400

    ext = os.path.splitext(f.filename)[1] or ('.jpg' if is_image else '.mp4')
    media_path = os.path.join(UPLOAD_DIR, f'{uuid.uuid4().hex}{ext}')
    f.save(media_path)

    def generate():
        try:
            for tok in engine.infer_stream(media_path, is_image, question):
                yield f'data:{tok}\n\n'
        except Exception as e:
            yield f'data:__error__{e}\n\n'
        finally:
            try: os.remove(media_path)
            except: pass

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'X-Accel-Buffering': 'no',
                             'Cache-Control': 'no-cache'})

if __name__ == '__main__':
    print('VILA Web Service → http://0.0.0.0:5003')
    app.run(host='0.0.0.0', port=5003, threaded=False)
