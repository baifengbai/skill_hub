import os
import sys
import json
import time
import threading
import tempfile
import numpy as np

DEMO_DIR   = '/data/LLM-TPU/models/Qwen3_5/python_demo'
CONFIG_DIR = '/data/LLM-TPU/models/Qwen3_5/config'
MODEL_DIR  = '/data/LLM-TPU/models/Qwen3_5/models/BM1684X'
UPLOAD_DIR = '/tmp/qwen3_5_uploads'
PORT       = 5004

sys.path.insert(0, DEMO_DIR)
os.chdir(DEMO_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)

from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context

app = Flask(__name__)

# ── 全局模型 ─────────────────────────────────────────────────────────────────
model        = None
model_status = 'loading'
model_error  = ''
model_lock   = threading.Lock()


def _find_bmodel():
    for f in sorted(os.listdir(MODEL_DIR)):
        if f.endswith('.bmodel') and 'bm1684x' in f:
            return os.path.join(MODEL_DIR, f)
    raise FileNotFoundError(f'No BM1684X bmodel found in {MODEL_DIR}')


def _load_model():
    global model, model_status, model_error
    try:
        print('[Qwen3.5] 加载模型中...', flush=True)
        bmodel_path = _find_bmodel()
        print(f'[Qwen3.5] bmodel: {bmodel_path}', flush=True)

        from pipeline import Qwen3_5

        class _Args:
            devid       = 0
            model_path  = bmodel_path
            config_path = CONFIG_DIR
            video_ratio = 0.25

        model = Qwen3_5(_Args())
        model_status = 'ready'
        print('[Qwen3.5] 模型加载完成', flush=True)
    except Exception as e:
        import traceback
        model_error  = str(e) + '\n' + traceback.format_exc()
        model_status = 'error'
        print(f'[Qwen3.5] 加载失败: {e}', flush=True)


threading.Thread(target=_load_model, daemon=True).start()

# ── 推理核心（流式生成器）────────────────────────────────────────────────────

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}


def _detect_media(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return 'image'
    if ext in VIDEO_EXTS:
        return 'video'
    return None


def _stream_infer(question, media_path=None):
    """Yield SSE lines: data:<json>\n\n"""
    import torch

    media_type = _detect_media(media_path) if media_path else 'text'
    model.input_str = question

    if media_type == 'image':
        messages = model.image_message(media_path)
    elif media_type == 'video':
        messages = model.video_message(media_path)
    else:
        messages = model.text_message()

    inputs = model.process(messages, media_type)
    token_len = inputs.input_ids.numel()

    if token_len > model.model.MAX_INPUT_LENGTH:
        yield f'data:{json.dumps("__error__输入过长: " + str(token_len) + " > " + str(model.model.MAX_INPUT_LENGTH))}\n\n'
        return

    # prefill
    t0 = time.time()
    model.model.forward_embed(inputs.input_ids.numpy())

    if media_type == 'image':
        model.vit_process_image(inputs)
        position_ids = model.get_rope_index(inputs.input_ids, inputs.image_grid_thw, model.ID_IMAGE_PAD)
        model.max_posid = int(position_ids.max())
        token = model.forward_prefill(position_ids.numpy())
    elif media_type == 'video':
        model.vit_process_video(inputs)
        position_ids = model.get_rope_index(inputs.input_ids, inputs.video_grid_thw, model.ID_VIDEO_PAD)
        model.max_posid = int(position_ids.max())
        token = model.forward_prefill(position_ids.numpy())
    else:
        position_ids = 3 * [i for i in range(token_len)]
        model.max_posid = token_len - 1
        token = model.forward_prefill(np.array(position_ids, dtype=np.int32))

    t_prefill = time.time() - t0
    yield f'data:{json.dumps({"type": "perf", "prefill": round(t_prefill, 2)})}\n\n'

    # decode
    full_word_tokens = []
    text = ''
    tok_num = 0

    while token not in [model.ID_IM_END] and model.model.history_length < model.model.SEQLEN:
        full_word_tokens.append(token)
        word = model.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
        if '\ufffd' not in word:
            if len(full_word_tokens) == 1:
                pre_word = word
                word = model.tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
            text += word
            yield f'data:{json.dumps(text)}\n\n'
            full_word_tokens = []

        model.max_posid += 1
        position_ids = np.array([model.max_posid, model.max_posid, model.max_posid], dtype=np.int32)
        token = model.model.forward_next(position_ids)
        tok_num += 1

    model.history_max_posid = model.max_posid + 2
    t_total = time.time() - t0
    tps = tok_num / (t_total - t_prefill) if t_total > t_prefill else 0

    yield f'data:{json.dumps({"type": "perf", "total": round(t_total, 2), "tokens": tok_num, "tps": round(tps, 1)})}\n\n'
    yield 'data:"__done__"\n\n'


# ── HTML ─────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3.5 VL — 多模态对话</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#0d1117;color:#e6edf3;min-height:100vh;display:flex;flex-direction:column}
.header{background:rgba(255,255,255,.03);border-bottom:1px solid rgba(255,255,255,.07);
  padding:14px 24px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.header h1{font-size:17px;font-weight:700}
.badge{padding:3px 10px;border-radius:10px;font-size:11px;border:1px solid}
.badge.ready{background:rgba(46,160,67,.15);color:#3fb950;border-color:rgba(46,160,67,.3)}
.badge.loading{background:rgba(255,165,0,.15);color:#ffa500;border-color:rgba(255,165,0,.3)}
.badge.error{background:rgba(239,68,68,.15);color:#fca5a5;border-color:rgba(239,68,68,.3)}
.header-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.btn-sm{padding:5px 14px;border-radius:8px;border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.06);color:#8b949e;font-size:12px;cursor:pointer;transition:.15s}
.btn-sm:hover{color:#e6edf3;border-color:rgba(255,255,255,.25)}
.chat-wrap{flex:1;display:flex;flex-direction:column;overflow:hidden;max-width:900px;
  width:100%;margin:0 auto;padding:0 16px}
.chat-scroll{flex:1;overflow-y:auto;padding:24px 0;display:flex;flex-direction:column;gap:14px}
.msg{display:flex;flex-direction:column;gap:4px}
.msg.user{align-items:flex-end}
.msg.assistant{align-items:flex-start}
.msg-label{font-size:11px;color:#484f58;padding:0 4px}
.msg-body{padding:11px 16px;border-radius:12px;font-size:14px;line-height:1.7;
  white-space:pre-wrap;word-break:break-word;max-width:82%}
.msg.user .msg-body{background:rgba(88,166,255,.12);border:1px solid rgba(88,166,255,.2)}
.msg.assistant .msg-body{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08)}
.msg-media{max-width:320px;max-height:240px;border-radius:8px;margin-bottom:6px}
.perf{font-size:11px;color:#484f58;padding:0 4px}
.typing{display:inline-block;width:7px;height:13px;background:#58a6ff;
  border-radius:2px;animation:blink .7s infinite;vertical-align:middle;margin-left:2px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.empty-hint{text-align:center;color:#484f58;padding:60px 20px;font-size:14px;flex:1;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px}
.input-area{border-top:1px solid rgba(255,255,255,.07);padding:16px 0 20px}
.input-row{display:flex;gap:10px;align-items:flex-end}
textarea{flex:1;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);
  border-radius:12px;padding:11px 14px;color:#e6edf3;font-size:14px;resize:none;
  min-height:52px;max-height:180px;font-family:inherit;line-height:1.5}
textarea:focus{outline:none;border-color:rgba(88,166,255,.4)}
.send-btn{padding:11px 20px;background:linear-gradient(135deg,#238636,#2ea043);
  border:none;border-radius:10px;color:#fff;font-size:14px;font-weight:600;
  cursor:pointer;transition:.2s;white-space:nowrap;height:52px}
.send-btn:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 4px 16px rgba(46,160,67,.3)}
.send-btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
.file-row{display:flex;gap:8px;align-items:center;margin-bottom:8px}
.file-label{padding:6px 14px;border-radius:8px;border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.06);color:#8b949e;font-size:12px;cursor:pointer;transition:.15s}
.file-label:hover{color:#e6edf3;border-color:rgba(255,255,255,.25)}
.file-name{font-size:12px;color:#58a6ff;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.file-clear{font-size:11px;color:#f85149;cursor:pointer;text-decoration:underline}
.hint{font-size:11px;color:#484f58;margin-top:6px;padding:0 2px}
.overlay{position:fixed;inset:0;background:rgba(13,17,23,.92);display:flex;
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
  <h1>Qwen3.5 VL <span class="badge loading" id="statusBadge">加载中</span></h1>
  <span style="color:#484f58;font-size:12px">2B int4 · seq2048 · BM1684X</span>
  <div class="header-right">
    <button class="btn-sm" onclick="clearHistory()">清空对话</button>
  </div>
</div>

<div class="chat-wrap">
  <div class="chat-scroll" id="chatScroll">
    <div class="empty-hint" id="emptyHint">
      <div style="font-size:42px">🖼️</div>
      <div>Qwen3.5 VL 多模态模型</div>
      <div style="font-size:12px;color:#484f58;margin-top:4px">支持图片/视频问答 · 也可纯文字对话 · seq_len=2048</div>
    </div>
  </div>

  <div class="input-area">
    <div class="file-row">
      <label class="file-label">
        📎 上传图片/视频
        <input type="file" id="fileInput" accept="image/*,video/*" style="display:none" onchange="onFileSelect(this)">
      </label>
      <span class="file-name" id="fileName"></span>
      <span class="file-clear" id="fileClear" style="display:none" onclick="clearFile()">清除</span>
    </div>
    <div class="input-row">
      <textarea id="inputBox" rows="1" placeholder="输入问题（可选择上传图片/视频），Shift+Enter 换行，Enter 发送..."
        oninput="autoResize(this)" onkeydown="onKey(event)"></textarea>
      <button class="send-btn" id="sendBtn" onclick="sendMsg()" disabled>发送</button>
    </div>
    <div class="hint">图片 token ≈ 长×宽÷32÷32 · 视频默认 1fps，按图片 1/4 尺寸</div>
  </div>
</div>

<script>
let isReady=false, isGenerating=false, selectedFile=null;

// ── 状态轮询 ─────────────────────────────────────────────────────────────────
(async function poll(){
  try{
    const d=await(await fetch('/api/status')).json();
    const badge=document.getElementById('statusBadge');
    badge.className='badge '+d.status;
    badge.textContent={loading:'加载中',ready:'就绪',error:'加载失败'}[d.status]||d.status;
    if(d.status==='ready'){
      isReady=true;
      document.getElementById('overlay').classList.add('hide');
      document.getElementById('sendBtn').disabled=false;
    }else if(d.status==='error'){
      document.getElementById('overlayMsg').textContent='加载失败: '+d.error;
      document.getElementById('overlay').querySelector('.spinner').style.display='none';
    }else{setTimeout(poll,3000);}
  }catch(e){setTimeout(poll,3000);}
})();

// ── 文件选择 ─────────────────────────────────────────────────────────────────
function onFileSelect(input){
  if(input.files.length){
    selectedFile=input.files[0];
    document.getElementById('fileName').textContent=selectedFile.name;
    document.getElementById('fileClear').style.display='inline';
  }
}
function clearFile(){
  selectedFile=null;
  document.getElementById('fileInput').value='';
  document.getElementById('fileName').textContent='';
  document.getElementById('fileClear').style.display='none';
}

// ── 发送 ─────────────────────────────────────────────────────────────────────
function onKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg();}}
function autoResize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,180)+'px';}

async function sendMsg(){
  if(!isReady||isGenerating)return;
  const q=document.getElementById('inputBox').value.trim();
  if(!q)return;

  document.getElementById('inputBox').value='';
  document.getElementById('inputBox').style.height='auto';
  document.getElementById('sendBtn').disabled=true;
  document.getElementById('emptyHint')?.remove();
  isGenerating=true;

  // 显示用户消息
  const userBody=addMsg('user',q);
  if(selectedFile){
    const preview=document.createElement(selectedFile.type.startsWith('video/')?'video':'img');
    preview.className='msg-media';
    preview.src=URL.createObjectURL(selectedFile);
    if(preview.tagName==='VIDEO'){preview.controls=true;preview.muted=true;}
    userBody.prepend(preview);
  }

  const bodyEl=addMsg('assistant','',true);
  const t0=Date.now();

  // 构建请求
  const fd=new FormData();
  fd.append('message',q);
  if(selectedFile)fd.append('media',selectedFile);

  try{
    const resp=await fetch('/api/chat',{method:'POST',body:fd});
    if(!resp.ok){
      const err=await resp.json().catch(()=>({error:'HTTP '+resp.status}));
      bodyEl.innerHTML=escHtml('[错误] '+(err.error||resp.status));
      throw {handled:true};
    }
    const reader=resp.body.getReader();
    const dec=new TextDecoder();
    let buf='',finished=false;

    while(!finished){
      const{done,value}=await reader.read();
      if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\n');
      buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data:'))continue;
        let tok;try{tok=JSON.parse(line.slice(5));}catch(e){continue;}
        if(tok==='__done__'){finished=true;break;}
        if(typeof tok==='string'&&tok.startsWith('__error__')){
          bodyEl.innerHTML=escHtml('[错误] '+tok.slice(9));
          finished=true;break;
        }
        if(typeof tok==='object'&&tok.type==='perf'){
          // 性能数据在最后展示
          if(tok.tps){
            const perf=document.createElement('div');
            perf.className='perf';
            perf.textContent=`prefill ${tok.total}s · ${tok.tokens} tokens · ${tok.tps} tok/s`;
            bodyEl.parentNode.appendChild(perf);
          }
          continue;
        }
        // 累积文本
        if(typeof tok==='string'){
          let mainSpan=bodyEl.querySelector('.main-text')||bodyEl;
          mainSpan.innerHTML=escHtml(tok)+'<span class="typing"></span>';
          document.getElementById('chatScroll').scrollTop=999999;
        }
      }
    }
    try{await reader.cancel();}catch(e){}
  }catch(e){
    if(!e.handled) bodyEl.innerHTML=escHtml('[网络错误] '+(e.message||''));
  }

  bodyEl.querySelector('.typing')?.remove();
  if(!bodyEl.parentNode.querySelector('.perf')){
    const perf=document.createElement('div');
    perf.className='perf';
    perf.textContent=`耗时 ${((Date.now()-t0)/1000).toFixed(1)}s`;
    bodyEl.parentNode.appendChild(perf);
  }

  clearFile();
  isGenerating=false;
  document.getElementById('sendBtn').disabled=false;
  document.getElementById('chatScroll').scrollTop=999999;
}

function addMsg(role,text,withCursor=false){
  const scroll=document.getElementById('chatScroll');
  const div=document.createElement('div');
  div.className='msg '+role;
  const label=role==='user'?'👤 你':'🤖 Qwen3.5';
  const body=document.createElement('div');
  body.className='msg-body';
  if(text)body.innerHTML=escHtml(text);
  if(withCursor)body.innerHTML='<span class="typing"></span>';
  div.innerHTML=`<div class="msg-label">${label}</div>`;
  div.appendChild(body);
  scroll.appendChild(div);
  scroll.scrollTop=999999;
  return body;
}

async function clearHistory(){
  await fetch('/api/clear',{method:'POST'});
  const scroll=document.getElementById('chatScroll');
  scroll.innerHTML=`<div class="empty-hint" id="emptyHint">
    <div style="font-size:42px">🖼️</div><div>对话已清空，重新开始吧</div></div>`;
}

function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
</script>
</body>
</html>"""

# ── Flask 路由 ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/status')
def api_status():
    return jsonify({'status': model_status,
                    'error': model_error[:500] if model_error else ''})


@app.route('/api/chat', methods=['POST'])
def api_chat():
    if model_status != 'ready':
        return jsonify({'error': '模型未就绪'}), 503

    message = request.form.get('message', '').strip()
    if not message:
        return jsonify({'error': '消息不能为空'}), 400

    # 保存上传的媒体文件
    media_path = None
    media_file = request.files.get('media')
    if media_file and media_file.filename:
        ext = os.path.splitext(media_file.filename)[1].lower()
        tmp = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=ext, delete=False)
        media_file.save(tmp.name)
        media_path = tmp.name

    def generate():
        with model_lock:
            try:
                yield from _stream_infer(message, media_path)
            except Exception as e:
                yield f'data:{json.dumps("__error__" + str(e))}\n\n'
            finally:
                # 单轮模式：清空历史
                try:
                    model.model.clear_history()
                    model.history_max_posid = 0
                except Exception:
                    pass
                # 清理临时文件
                if media_path:
                    try:
                        os.unlink(media_path)
                    except Exception:
                        pass

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})


@app.route('/api/clear', methods=['POST'])
def api_clear():
    if model and model_status == 'ready':
        with model_lock:
            model.model.clear_history()
            model.history_max_posid = 0
    return jsonify({'ok': True})


if __name__ == '__main__':
    print(f'Qwen3.5 VL Web Service → http://0.0.0.0:{PORT}')
    app.run(host='0.0.0.0', port=PORT, threaded=False)
