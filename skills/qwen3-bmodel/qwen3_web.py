import os
import sys
import json
import threading
import time
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context

# ── 路径 ──────────────────────────────────────────────────────────────────────
DEMO_DIR   = '/data/LLM-TPU/models/Qwen3/python_demo'
MODEL_PATH = '/data/LLM-TPU/models/Qwen3/models/BM1684X/qwen3-4b-awq_w4bf16_seq512_bm1684x_1dev_20250514_161445.bmodel'
CONFIG_DIR = f'{DEMO_DIR}/config'

sys.path.insert(0, DEMO_DIR)
os.chdir(DEMO_DIR)

app = Flask(__name__)

# ── 全局模型实例（异步加载）───────────────────────────────────────────────────
model        = None
model_status = 'loading'
model_error  = ''
model_lock   = threading.Lock()

def _load_model():
    global model, model_status, model_error
    try:
        print('[Qwen3] 加载模型中...', flush=True)
        import chat
        from transformers import AutoTokenizer

        class _Args:
            devid           = '0'
            temperature     = 1.0
            top_p           = 1.0
            repeat_penalty  = 1.0
            repeat_last_n   = 32
            max_new_tokens  = 1024
            generation_mode = 'greedy'
            enable_history  = True
            config_path     = CONFIG_DIR
            model_path      = MODEL_PATH

        from pipeline import Qwen2
        model = Qwen2(_Args())
        model_status = 'ready'
        print('[Qwen3] 模型加载完成', flush=True)
    except Exception as e:
        import traceback
        model_error  = str(e) + '\n' + traceback.format_exc()
        model_status = 'error'
        print(f'[Qwen3] 加载失败: {e}', flush=True)

threading.Thread(target=_load_model, daemon=True).start()

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-4B — 对话</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#0d1117;color:#e6edf3;min-height:100vh;display:flex;flex-direction:column}
.header{background:rgba(255,255,255,.03);border-bottom:1px solid rgba(255,255,255,.07);
  padding:14px 24px;display:flex;align-items:center;gap:12px}
.header h1{font-size:17px;font-weight:700}
.badge{padding:3px 10px;border-radius:10px;font-size:11px;border:1px solid}
.badge.ready{background:rgba(46,160,67,.15);color:#3fb950;border-color:rgba(46,160,67,.3)}
.badge.loading{background:rgba(255,165,0,.15);color:#ffa500;border-color:rgba(255,165,0,.3)}
.badge.error{background:rgba(239,68,68,.15);color:#fca5a5;border-color:rgba(239,68,68,.3)}
.header-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.btn-sm{padding:5px 14px;border-radius:8px;border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.06);color:#8b949e;font-size:12px;cursor:pointer;transition:.15s}
.btn-sm:hover{color:#e6edf3;border-color:rgba(255,255,255,.25)}
.chat-wrap{flex:1;display:flex;flex-direction:column;overflow:hidden;max-width:860px;
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
.think-block{font-size:12px;color:#6e7681;border-left:2px solid #30363d;
  padding:6px 10px;margin-bottom:6px;white-space:pre-wrap;word-break:break-word}
.perf{font-size:11px;color:#484f58;padding:0 4px}
.typing{display:inline-block;width:7px;height:13px;background:#58a6ff;
  border-radius:2px;animation:blink .7s infinite;vertical-align:middle;margin-left:2px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.empty-hint{text-align:center;color:#484f58;padding:80px 20px;font-size:14px;flex:1;
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
.hint{font-size:11px;color:#484f58;margin-top:6px;padding:0 2px}
/* 加载遮罩 */
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
  <h1>Qwen3-4B <span class="badge loading" id="statusBadge">加载中</span></h1>
  <span style="color:#484f58;font-size:12px">seq512 · w4bf16 · BM1684X</span>
  <div class="header-right">
    <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:#8b949e;cursor:pointer">
      <input type="checkbox" id="thinkToggle" onchange="toggleThink(this)"
        style="width:14px;height:14px;accent-color:#58a6ff">
      思考模式
    </label>
    <button class="btn-sm" onclick="clearHistory()">清空对话</button>
  </div>
</div>

<div class="chat-wrap">
  <div class="chat-scroll" id="chatScroll">
    <div class="empty-hint" id="emptyHint">
      <div style="font-size:42px">🤖</div>
      <div>Qwen3-4B 已就绪，开始对话吧</div>
      <div style="font-size:12px;color:#484f58;margin-top:4px">支持多轮对话 · seq_len=512，建议关闭思考模式（默认已关闭）</div>
    </div>
  </div>

  <div class="input-area">
    <div class="input-row">
      <textarea id="inputBox" rows="1" placeholder="输入问题，Shift+Enter 换行，Enter 发送..."
        oninput="autoResize(this)" onkeydown="onKey(event)"></textarea>
      <button class="send-btn" id="sendBtn" onclick="sendMsg()" disabled>发送</button>
    </div>
    <div class="hint">提示：首次推理需预热 KV 缓存，稍慢；后续响应更快</div>
  </div>
</div>

<script>
let isReady = false, isGenerating = false, thinkMode = false;

function toggleThink(cb){ thinkMode = cb.checked; }

// ── 状态轮询 ─────────────────────────────────────────────────────────────────
(async function poll(){
  try{
    const d = await (await fetch('/api/status')).json();
    const badge = document.getElementById('statusBadge');
    badge.className = 'badge ' + d.status;
    badge.textContent = {loading:'加载中', ready:'就绪', error:'加载失败'}[d.status] || d.status;
    if(d.status === 'ready'){
      isReady = true;
      document.getElementById('overlay').classList.add('hide');
      document.getElementById('sendBtn').disabled = false;
    } else if(d.status === 'error'){
      document.getElementById('overlayMsg').textContent = '加载失败: ' + d.error;
      document.getElementById('overlay').querySelector('.spinner').style.display = 'none';
    } else {
      setTimeout(poll, 3000);
    }
  } catch(e){ setTimeout(poll, 3000); }
})();

// ── 发送 ─────────────────────────────────────────────────────────────────────
function onKey(e){
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendMsg(); }
}

function autoResize(el){
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

async function sendMsg(){
  if(!isReady || isGenerating) return;
  const q = document.getElementById('inputBox').value.trim();
  if(!q) return;

  document.getElementById('inputBox').value = '';
  document.getElementById('inputBox').style.height = 'auto';
  document.getElementById('sendBtn').disabled = true;
  document.getElementById('emptyHint')?.remove();
  isGenerating = true;

  addMsg('user', q);

  const bodyEl = addMsg('assistant', '', true);
  const t0 = Date.now();

  let prevLen = 0, thinkEl = null, mainEl = bodyEl;

  try{
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: thinkMode ? q : q + ' /no_think'})
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    let finished = false;
    while(!finished){
      const {done, value} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for(const line of lines){
        if(!line.startsWith('data:')) continue;
        let tok; try{ tok = JSON.parse(line.slice(5)); } catch(e){ continue; }
        if(tok === '__done__'){ finished = true; break; }
        if(typeof tok === 'string' && tok.startsWith('__error__')){
          bodyEl.innerHTML = escHtml('[错误] ' + tok.slice(9));
          finished = true; break;
        }
        // tok 是当前完整输出（累积）
        renderOutput(tok, bodyEl);
      }
    }
    try{ await reader.cancel(); }catch(e){}
  } catch(e){
    bodyEl.innerHTML = escHtml('[网络错误] ' + e.message);
  }

  // 移除打字光标
  bodyEl.querySelector('.typing')?.remove();
  const perf = document.createElement('div');
  perf.className = 'perf';
  perf.textContent = `耗时 ${((Date.now()-t0)/1000).toFixed(1)}s`;
  bodyEl.parentNode.appendChild(perf);

  isGenerating = false;
  document.getElementById('sendBtn').disabled = false;
  document.getElementById('chatScroll').scrollTop = 999999;
}

function renderOutput(text, bodyEl){
  // 分离 <think>...</think> 思考块与正文
  const thinkMatch = text.match(/^<think>([\s\S]*?)<\/think>([\s\S]*)$/);
  if(thinkMatch){
    let thinkEl = bodyEl.querySelector('.think-block');
    if(!thinkEl){
      thinkEl = document.createElement('div');
      thinkEl.className = 'think-block';
      bodyEl.prepend(thinkEl);
    }
    thinkEl.textContent = thinkMatch[1].trim();
    const main = thinkMatch[2].trim();
    let mainSpan = bodyEl.querySelector('.main-text');
    if(!mainSpan){
      mainSpan = document.createElement('span');
      mainSpan.className = 'main-text';
      bodyEl.appendChild(mainSpan);
    }
    mainSpan.innerHTML = escHtml(main) + '<span class="typing"></span>';
  } else {
    // 思考未结束或无思考块，直接显示
    const thinking = text.startsWith('<think>');
    let mainSpan = bodyEl.querySelector('.main-text') || bodyEl;
    mainSpan.innerHTML = (thinking ? '<em style="color:#6e7681">' + escHtml(text) + '</em>' : escHtml(text)) + '<span class="typing"></span>';
  }
  document.getElementById('chatScroll').scrollTop = 999999;
}

function addMsg(role, text, withCursor=false){
  const scroll = document.getElementById('chatScroll');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const label = role === 'user' ? '👤 你' : '🤖 Qwen3';
  const body = document.createElement('div');
  body.className = 'msg-body';
  if(text) body.innerHTML = escHtml(text);
  if(withCursor) body.innerHTML = '<span class="typing"></span>';
  div.innerHTML = `<div class="msg-label">${label}</div>`;
  div.appendChild(body);
  scroll.appendChild(div);
  scroll.scrollTop = 999999;
  return body;
}

async function clearHistory(){
  await fetch('/api/clear', {method:'POST'});
  const scroll = document.getElementById('chatScroll');
  scroll.innerHTML = `<div class="empty-hint" id="emptyHint">
    <div style="font-size:42px">🤖</div>
    <div>对话已清空，重新开始吧</div>
  </div>`;
}

function escHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
</script>
</body>
</html>"""

# ── Flask 路由 ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/status')
def api_status():
    return jsonify({'status': model_status,
                    'error': model_error[:300] if model_error else ''})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if model_status != 'ready':
        return jsonify({'error': '模型未就绪'}), 503
    data    = request.get_json()
    message = (data or {}).get('message', '').strip()
    if not message:
        return jsonify({'error': '消息不能为空'}), 400

    def generate():
        with model_lock:
            try:
                prev = ''
                for answer, _ in model.stream_predict(message):
                    if answer != prev:
                        prev = answer
                        # 去掉 /no_think 指令本身（不显示给用户）
                        display = answer.replace(' /no_think', '')
                        yield f'data:{json.dumps(display)}\n\n'
                yield 'data:"__done__"\n\n'
            except Exception as e:
                yield f'data:{json.dumps("__error__" + str(e))}\n\n'
            finally:
                # 单轮模式：每次对话后清空历史，避免 seq512 被累积占满
                try:
                    model.clear()
                except Exception:
                    pass

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})

@app.route('/api/clear', methods=['POST'])
def api_clear():
    if model and model_status == 'ready':
        with model_lock:
            model.clear()
    return jsonify({'ok': True})

if __name__ == '__main__':
    print('Qwen3 Web Service → http://0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000, threaded=False)
