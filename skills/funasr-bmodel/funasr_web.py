import os
import sys
import uuid
import tempfile
import subprocess
sys.path.insert(0, '/data/FunASR-bmodel')

from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
UPLOAD_DIR = '/tmp/funasr_uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 全局模型，启动时加载一次
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
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FunASR 语音识别</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }
  .card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 40px;
    width: 100%;
    max-width: 640px;
    color: #fff;
  }
  h1 { font-size: 24px; font-weight: 700; margin-bottom: 6px; }
  .subtitle { color: rgba(255,255,255,0.5); font-size: 14px; margin-bottom: 32px; }
  .record-btn {
    width: 100%;
    padding: 18px;
    border-radius: 16px;
    border: none;
    font-size: 18px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
  }
  .record-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(102,126,234,0.4); }
  .record-btn.recording {
    background: linear-gradient(135deg, #f093fb, #f5576c);
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(245,87,108,0.4); }
    50% { box-shadow: 0 0 0 12px rgba(245,87,108,0); }
  }
  .or-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 20px 0;
    color: rgba(255,255,255,0.3);
    font-size: 13px;
  }
  .or-divider::before, .or-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.1);
  }
  .upload-area {
    border: 2px dashed rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    color: rgba(255,255,255,0.5);
    font-size: 14px;
  }
  .upload-area:hover { border-color: rgba(102,126,234,0.6); color: rgba(255,255,255,0.8); }
  .upload-area input { display: none; }
  .status {
    margin-top: 20px;
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 14px;
    display: none;
  }
  .status.show { display: block; }
  .status.info { background: rgba(102,126,234,0.2); color: #a5b4fc; }
  .status.processing { background: rgba(245,158,11,0.15); color: #fcd34d; }
  .status.error { background: rgba(239,68,68,0.15); color: #fca5a5; }
  .result-box {
    margin-top: 20px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    display: none;
  }
  .result-box.show { display: block; }
  .result-label { font-size: 12px; color: rgba(255,255,255,0.4); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
  .result-text { font-size: 18px; line-height: 1.7; color: #e2e8f0; }
  .time-badge {
    display: inline-block;
    margin-top: 12px;
    padding: 4px 10px;
    background: rgba(102,126,234,0.2);
    border-radius: 20px;
    font-size: 12px;
    color: #a5b4fc;
  }
  .history { margin-top: 24px; }
  .history-title { font-size: 13px; color: rgba(255,255,255,0.4); margin-bottom: 10px; }
  .history-item {
    padding: 10px 14px;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 14px;
    color: rgba(255,255,255,0.7);
    border-left: 3px solid rgba(102,126,234,0.5);
  }
  audio { width: 100%; margin-top: 12px; border-radius: 8px; }
</style>
</head>
<body>
<div class="card">
  <h1>🎙 FunASR 语音识别</h1>
  <p class="subtitle">BM1684X TPU 加速 · Paraformer 大模型</p>

  <button class="record-btn" id="recordBtn" onclick="toggleRecord()">按住录音</button>

  <div class="or-divider">或上传音频文件</div>

  <div class="upload-area" onclick="document.getElementById('fileInput').click()">
    <input type="file" id="fileInput" accept="audio/*" onchange="uploadFile(this)">
    📂 点击选择 WAV / MP3 / M4A 文件
    <div id="fileName" style="margin-top:6px;color:#a5b4fc;"></div>
  </div>

  <audio id="audioPlayer" controls style="display:none"></audio>

  <div class="status" id="status"></div>

  <div class="result-box" id="resultBox">
    <div class="result-label">识别结果</div>
    <div class="result-text" id="resultText"></div>
    <div class="time-badge" id="timeBadge"></div>
  </div>

  <div class="history" id="historyDiv" style="display:none">
    <div class="history-title">历史记录</div>
    <div id="historyList"></div>
  </div>
</div>

<script>
let mediaRecorder, audioChunks = [], isRecording = false;
const history = [];

function setStatus(msg, type='info') {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status show ' + type;
}

async function toggleRecord() {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, sampleRate: 16000 });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        const player = document.getElementById('audioPlayer');
        player.src = url;
        player.style.display = 'block';
        recognize(blob, 'recording.webm');
      };
      mediaRecorder.start();
      isRecording = true;
      document.getElementById('recordBtn').textContent = '⏹ 点击停止录音';
      document.getElementById('recordBtn').classList.add('recording');
      setStatus('录音中...', 'info');
    } catch(e) {
      setStatus('麦克风权限被拒绝: ' + e.message, 'error');
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;
    document.getElementById('recordBtn').textContent = '按住录音';
    document.getElementById('recordBtn').classList.remove('recording');
  }
}

function uploadFile(input) {
  if (!input.files.length) return;
  const file = input.files[0];
  document.getElementById('fileName').textContent = file.name;
  const url = URL.createObjectURL(file);
  const player = document.getElementById('audioPlayer');
  player.src = url;
  player.style.display = 'block';
  recognize(file, file.name);
}

async function recognize(blob, filename) {
  setStatus('上传并识别中，请稍候...', 'processing');
  document.getElementById('resultBox').classList.remove('show');
  const fd = new FormData();
  fd.append('audio', blob, filename);
  try {
    const t0 = Date.now();
    const res = await fetch('/api/recognize', { method: 'POST', body: fd });
    const data = await res.json();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    if (data.error) { setStatus('识别失败: ' + data.error, 'error'); return; }
    document.getElementById('resultText').textContent = data.text || '（无结果）';
    document.getElementById('timeBadge').textContent = `推理耗时 ${data.inference_time}s · 总耗时 ${elapsed}s`;
    document.getElementById('resultBox').classList.add('show');
    setStatus('识别完成', 'info');
    addHistory(data.text);
  } catch(e) {
    setStatus('请求失败: ' + e.message, 'error');
  }
}

function addHistory(text) {
  if (!text) return;
  history.unshift(text);
  if (history.length > 5) history.pop();
  const list = document.getElementById('historyList');
  list.innerHTML = history.map(t => `<div class="history-item">${t}</div>`).join('');
  document.getElementById('historyDiv').style.display = 'block';
}
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/recognize', methods=['POST'])
def recognize():
    if model is None:
        return jsonify({'error': '模型未加载'}), 503

    if 'audio' not in request.files:
        return jsonify({'error': '未收到音频文件'}), 400

    f = request.files['audio']
    ext = os.path.splitext(f.filename)[1] or '.webm'
    raw_path = os.path.join(UPLOAD_DIR, f'{uuid.uuid4()}{ext}')
    wav_path = raw_path.replace(ext, '.wav')

    try:
        f.save(raw_path)

        # 转换为 16kHz 单声道 WAV
        subprocess.run([
            'ffmpeg', '-y', '-i', raw_path,
            '-ar', '16000', '-ac', '1', '-f', 'wav', wav_path
        ], capture_output=True, check=True)

        import time
        t0 = time.time()
        res = model.generate(input=wav_path, batch_size_s=300)
        elapsed = round(time.time() - t0, 2)

        text = res[0].get('text', '') if res else ''
        return jsonify({'text': text, 'inference_time': elapsed})

    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'音频转换失败: {e.stderr.decode()}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        for p in [raw_path, wav_path]:
            try: os.remove(p)
            except: pass

if __name__ == '__main__':
    os.chdir('/data/FunASR-bmodel/bmodel')
    load_model()
    print('FunASR Web Service started at http://0.0.0.0:5001')
    app.run(host='0.0.0.0', port=5001, threaded=False)
