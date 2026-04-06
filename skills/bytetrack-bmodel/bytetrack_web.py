import os
import sys
import time
import uuid
import threading
import subprocess
import json
import cv2
import numpy as np
import sophon.sail as sail
from flask import Flask, request, jsonify, render_template_string, Response, send_file

# ── 路径设置 ──────────────────────────────────────────────────────────────────
BASE = '/data/sophon-demo/sample/ByteTrack/python'
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'detector/yolov5'))
os.chdir(BASE)

from yolov5_opencv import YOLOv5
from tracker.byte_tracker import ByteTracker
from tracker.utils.parser import get_config

BMODEL   = '/data/sophon-demo/sample/ByteTrack/models/BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel'
CFG_FILE = '/data/sophon-demo/sample/ByteTrack/python/configs/bytetrack.yaml'
DEMO_VIDEO = '/data/sophon-demo/sample/ByteTrack/datasets/test_car_person_1080P.mp4'
UPLOAD_DIR = '/tmp/bytetrack_uploads'
OUTPUT_DIR = '/tmp/bytetrack_output'
FRAMES_DIR = '/tmp/bytetrack_frames'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# 预览帧缩放尺寸（节省存储，加速传输）
PREVIEW_W, PREVIEW_H = 960, 540

app = Flask(__name__)

# ── 颜色表（按 track_id 着色）────────────────────────────────────────────────
np.random.seed(42)
TRACK_COLORS = np.random.randint(50, 255, size=(1000, 3), dtype=np.uint8).tolist()

def get_color(track_id):
    return tuple(TRACK_COLORS[track_id % 1000])

# ── 任务状态管理 ──────────────────────────────────────────────────────────────
tasks = {}   # task_id -> {status, progress, result, error, stats, frame_count, fps}

def run_tracking(task_id, video_path, conf_thresh, nms_thresh, track_thresh):
    try:
        tasks[task_id]['status'] = 'running'

        frame_dir = os.path.join(FRAMES_DIR, task_id)
        os.makedirs(frame_dir, exist_ok=True)

        # 加载检测器（YOLOv5 接受 args 对象）
        from types import SimpleNamespace
        det_args = SimpleNamespace(bmodel=BMODEL, dev_id=0,
                                   conf_thresh=conf_thresh, nms_thresh=nms_thresh)
        detector = YOLOv5(det_args)

        # 加载追踪器
        cfg = get_config()
        cfg.merge_from_file(CFG_FILE)
        tracker = ByteTracker(
            cfg.BYTETRACK.MIN_BOX_AREA,
            track_thresh,
            cfg.BYTETRACK.TRACK_BUFFER,
            cfg.BYTETRACK.MATCH_THRESH
        )

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25

        # 同时写 mp4v 文件供下载
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(OUTPUT_DIR, f'{task_id}.mp4')
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps_src, (w, h))

        frame_id = 0
        det_times, track_times = [], []
        max_track_ids = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            # 检测
            t0 = time.time()
            results = detector([frame])
            det_ms = (time.time() - t0) * 1000
            det_times.append(det_ms)

            det = results[0] if results else np.zeros((0, 6))
            bboxes = [[r[0], r[1], r[2], r[3]] for r in det if r[2]-r[0]>0 and r[3]-r[1]>0]
            confs  = [float(r[4]) for r in det if r[2]-r[0]>0 and r[3]-r[1]>0]
            clss   = [int(r[5])  for r in det if r[2]-r[0]>0 and r[3]-r[1]>0]

            # 追踪
            t1 = time.time()
            outputs = tracker._tracker_update(bboxes, confs, clss, frame)
            track_ms = (time.time() - t1) * 1000
            track_times.append(track_ms)

            # 绘制
            for val in outputs:
                x1, y1, bw, bh, cls_id, track_id = val
                x2, y2 = int(x1 + bw), int(y1 + bh)
                x1, y1 = int(x1), int(y1)
                color = get_color(track_id)
                max_track_ids.add(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'ID-{track_id} {cls_id}'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
                cv2.putText(frame, label, (x1+2, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

            # HUD
            avg_det = np.mean(det_times[-30:]) if det_times else 0
            avg_trk = np.mean(track_times[-30:]) if track_times else 0
            hud = f'Frame:{frame_id}  Det:{avg_det:.1f}ms  Track:{avg_trk:.1f}ms  Active:{len(outputs)}'
            cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0,255,0), 2, cv2.LINE_AA)

            # 写原始视频（供下载）
            writer.write(frame)

            # 保存预览帧（缩放 JPEG）
            preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
            cv2.imwrite(os.path.join(frame_dir, f'{frame_id:06d}.jpg'), preview,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])

            tasks[task_id]['progress'] = int(frame_id / max(total, 1) * 100)

        cap.release()
        writer.release()

        tasks[task_id].update({
            'status': 'done',
            'progress': 100,
            'result': out_path,
            'frame_count': frame_id,
            'fps': fps_src,
            'stats': {
                'frames': frame_id,
                'total_ids': len(max_track_ids),
                'avg_det_ms': round(float(np.mean(det_times)), 1) if det_times else 0,
                'avg_track_ms': round(float(np.mean(track_times)), 1) if track_times else 0,
                'avg_fps': round(1000 / max(float(np.mean(det_times)) + float(np.mean(track_times)), 1), 1) if det_times else 0,
            }
        })

    except Exception as e:
        import traceback
        tasks[task_id].update({'status': 'error', 'error': str(e),
                               'trace': traceback.format_exc()})

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ByteTrack — 多目标追踪</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#0d1117;color:#e6edf3;min-height:100vh;padding:24px}
.page{max-width:1200px;margin:0 auto}
h1{font-size:22px;font-weight:700;margin-bottom:4px}
.sub{color:#8b949e;font-size:13px;margin-bottom:28px}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;
  background:rgba(88,166,255,.15);color:#58a6ff;border:1px solid rgba(88,166,255,.3);margin-left:8px}
.grid{display:grid;grid-template-columns:340px 1fr;gap:20px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.panel{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
  border-radius:16px;padding:20px}
.panel-title{font-size:13px;color:#58a6ff;margin-bottom:16px;font-weight:600;
  text-transform:uppercase;letter-spacing:.05em}
.upload-zone{border:2px dashed rgba(88,166,255,.25);border-radius:12px;padding:28px;
  text-align:center;cursor:pointer;transition:.2s;color:#8b949e;font-size:13px}
.upload-zone:hover,.upload-zone.drag{border-color:#58a6ff;background:rgba(88,166,255,.06);color:#e6edf3}
.upload-zone input{display:none}
.upload-icon{font-size:32px;margin-bottom:8px}
.demo-btn{width:100%;margin-top:10px;padding:9px;background:rgba(88,166,255,.1);
  border:1px solid rgba(88,166,255,.25);border-radius:10px;color:#58a6ff;
  font-size:13px;cursor:pointer;transition:.2s}
.demo-btn:hover{background:rgba(88,166,255,.2)}
.file-name{margin-top:8px;font-size:12px;color:#58a6ff;word-break:break-all}
.param-label{font-size:12px;color:#8b949e;margin-bottom:5px;display:flex;justify-content:space-between}
.param-label span{color:#58a6ff;font-weight:600}
input[type=range]{width:100%;accent-color:#58a6ff;margin-bottom:12px}
.run-btn{width:100%;padding:13px;background:linear-gradient(135deg,#238636,#2ea043);
  border:none;border-radius:12px;color:#fff;font-size:15px;font-weight:600;
  cursor:pointer;transition:.2s;margin-top:4px}
.run-btn:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(46,160,67,.35)}
.run-btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
.progress-wrap{margin-top:14px;display:none}
.progress-wrap.show{display:block}
.progress-bar{height:6px;background:rgba(255,255,255,.08);border-radius:3px;overflow:hidden;margin-bottom:6px}
.progress-fill{height:100%;background:linear-gradient(90deg,#238636,#58a6ff);
  border-radius:3px;transition:width .3s;width:0%}
.progress-text{font-size:12px;color:#8b949e;text-align:center}
.stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px}
.stat-card{background:rgba(0,0,0,.3);border-radius:10px;padding:12px;text-align:center;
  border:1px solid rgba(255,255,255,.05)}
.stat-val{font-size:22px;font-weight:700;color:#58a6ff}
.stat-lbl{font-size:11px;color:#8b949e;margin-top:2px}
.video-wrap{background:#000;border-radius:12px;overflow:hidden;min-height:300px;
  display:flex;align-items:center;justify-content:center;position:relative;flex-direction:column}
#preview{width:100%;max-height:520px;object-fit:contain;display:none}
.video-placeholder{color:#484f58;text-align:center;padding:60px 20px;font-size:14px}
.error-box{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);
  border-radius:10px;padding:12px;color:#fca5a5;font-size:13px;margin-top:10px;display:none}
.player-ctrl{width:100%;padding:10px 14px;display:none;align-items:center;gap:10px;
  background:rgba(0,0,0,.4);border-top:1px solid rgba(255,255,255,.06)}
.player-ctrl.show{display:flex}
.play-btn{background:none;border:none;color:#fff;font-size:20px;cursor:pointer;padding:0;line-height:1}
#seekBar{flex:1;accent-color:#58a6ff;height:4px}
#timeLbl{font-size:12px;color:#8b949e;white-space:nowrap;min-width:80px;text-align:right}
.dl-btn{display:none;width:100%;margin-top:10px;padding:9px;
  background:rgba(56,139,253,.15);border:1px solid rgba(56,139,253,.3);
  border-radius:10px;color:#58a6ff;font-size:13px;cursor:pointer;text-decoration:none;text-align:center}
.dl-btn.show{display:block}
</style>
</head>
<body>
<div class="page">
  <h1>ByteTrack 多目标追踪<span class="badge">BM1684X fp16</span></h1>
  <p class="sub">YOLOv5s 检测 + ByteTrack 追踪 · TPU 加速推理</p>
  <div class="grid">
    <!-- 左栏：输入 + 参数 -->
    <div>
      <div class="panel" style="margin-bottom:16px">
        <div class="panel-title">视频输入</div>
        <div class="upload-zone" id="zone" onclick="document.getElementById('fi').click()">
          <div class="upload-icon">🎬</div>
          <div>点击或拖入视频文件</div>
          <small>MP4 / AVI / MOV</small>
          <input type="file" id="fi" accept="video/*" onchange="onFile(this)">
        </div>
        <div class="file-name" id="fname"></div>
        <button class="demo-btn" onclick="useDemo()">▶ 使用内置演示视频（行人+车辆 1080P）</button>
      </div>

      <div class="panel" style="margin-bottom:16px">
        <div class="panel-title">参数调节</div>
        <div class="param-label">置信度阈值 <span id="cv">0.40</span></div>
        <input type="range" id="conf" min="0.1" max="0.9" step="0.05" value="0.40"
               oninput="document.getElementById('cv').textContent=parseFloat(this.value).toFixed(2)">
        <div class="param-label">NMS 阈值 <span id="nv">0.70</span></div>
        <input type="range" id="nms" min="0.1" max="0.9" step="0.05" value="0.70"
               oninput="document.getElementById('nv').textContent=parseFloat(this.value).toFixed(2)">
        <div class="param-label">追踪阈值 <span id="tv">0.70</span></div>
        <input type="range" id="trk" min="0.1" max="0.9" step="0.05" value="0.70"
               oninput="document.getElementById('tv').textContent=parseFloat(this.value).toFixed(2)">
        <button class="run-btn" id="runBtn" onclick="startTask()">开始追踪</button>
        <div class="progress-wrap" id="progWrap">
          <div class="progress-bar"><div class="progress-fill" id="progFill"></div></div>
          <div class="progress-text" id="progText">处理中...</div>
        </div>
        <div class="error-box" id="errBox"></div>
      </div>

      <div class="panel" id="statsPanel" style="display:none">
        <div class="panel-title">推理性能</div>
        <div class="stats-grid" id="statsGrid"></div>
      </div>
    </div>

    <!-- 右栏：帧预览播放器 -->
    <div class="panel">
      <div class="panel-title">追踪结果</div>
      <div class="video-wrap" id="videoWrap">
        <div class="video-placeholder" id="vph">
          <div style="font-size:40px;margin-bottom:12px">🎯</div>
          选择视频并点击「开始追踪」<br>结果将在此处逐帧播放
        </div>
        <img id="preview" alt="追踪预览">
      </div>
      <div class="player-ctrl" id="playerCtrl">
        <button class="play-btn" id="playBtn" onclick="togglePlay()">⏸</button>
        <input type="range" id="seekBar" min="1" value="1" oninput="seekTo(this.value)">
        <span id="timeLbl">0 / 0</span>
      </div>
      <a class="dl-btn" id="dlBtn" href="#" download>⬇ 下载追踪结果视频（mp4v）</a>
    </div>
  </div>
</div>

<script>
let selectedFile = null;
let useDemo_ = false;
let pollTimer = null;
// 播放器状态
let frames = 0, fps = 25, curFrame = 1, playing = false, playTimer = null;
let taskId_ = null;

// 拖拽
const zone = document.getElementById('zone');
zone.ondragover = e => { e.preventDefault(); zone.classList.add('drag'); };
zone.ondragleave = () => zone.classList.remove('drag');
zone.ondrop = e => { e.preventDefault(); zone.classList.remove('drag');
  if(e.dataTransfer.files[0]) onFile_({files: e.dataTransfer.files}); };

function onFile(input){ onFile_(input); }
function onFile_(input){
  selectedFile = input.files[0]; useDemo_ = false;
  document.getElementById('fname').textContent = '📎 ' + selectedFile.name;
}
function useDemo(){
  selectedFile = null; useDemo_ = true;
  document.getElementById('fname').textContent = '📎 test_car_person_1080P.mp4（内置）';
}

async function startTask(){
  if(!selectedFile && !useDemo_){ alert('请先选择视频或使用演示视频'); return; }
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  document.getElementById('progWrap').classList.add('show');
  document.getElementById('errBox').style.display = 'none';
  document.getElementById('statsPanel').style.display = 'none';
  document.getElementById('preview').style.display = 'none';
  document.getElementById('vph').style.display = 'flex';
  document.getElementById('playerCtrl').classList.remove('show');
  document.getElementById('dlBtn').classList.remove('show');
  stopPlay();

  const fd = new FormData();
  if(selectedFile) fd.append('video', selectedFile);
  else fd.append('demo', '1');
  fd.append('conf_thresh', document.getElementById('conf').value);
  fd.append('nms_thresh', document.getElementById('nms').value);
  fd.append('track_thresh', document.getElementById('trk').value);

  try{
    const res = await fetch('/api/track', {method:'POST', body:fd});
    const {task_id} = await res.json();
    taskId_ = task_id;
    pollTimer = setInterval(() => poll(task_id, btn), 1500);
  }catch(e){
    showError(e.message); btn.disabled = false;
  }
}

async function poll(task_id, btn){
  try{
    const res = await fetch('/api/status/' + task_id);
    const d = await res.json();
    document.getElementById('progFill').style.width = d.progress + '%';
    document.getElementById('progText').textContent = d.status === 'done'
      ? '完成！' : `处理中... ${d.progress}%`;
    if(d.status === 'done'){
      clearInterval(pollTimer);
      btn.disabled = false;
      initPlayer(task_id, d.frame_count, d.fps, d.stats);
    } else if(d.status === 'error'){
      clearInterval(pollTimer);
      btn.disabled = false;
      showError(d.error);
    }
  }catch(e){ console.error(e); }
}

function initPlayer(task_id, fc, videoFps, stats){
  frames = fc; fps = videoFps || 25;
  curFrame = 1; playing = true;
  document.getElementById('vph').style.display = 'none';
  document.getElementById('preview').style.display = 'block';
  const seek = document.getElementById('seekBar');
  seek.max = frames; seek.value = 1;
  document.getElementById('playerCtrl').classList.add('show');
  const dlBtn = document.getElementById('dlBtn');
  dlBtn.href = '/api/result/' + task_id;
  dlBtn.classList.add('show');
  showStats(stats);
  startPlay(task_id);
}

function startPlay(task_id){
  stopPlay();
  playing = true;
  document.getElementById('playBtn').textContent = '⏸';
  const interval = Math.max(1000 / fps, 40);
  playTimer = setInterval(() => {
    if(curFrame > frames){ curFrame = 1; }
    document.getElementById('preview').src = '/api/frame/' + task_id + '/' + curFrame + '?t=' + Date.now();
    document.getElementById('seekBar').value = curFrame;
    document.getElementById('timeLbl').textContent = curFrame + ' / ' + frames;
    curFrame++;
  }, interval);
}

function stopPlay(){
  if(playTimer){ clearInterval(playTimer); playTimer = null; }
  playing = false;
  document.getElementById('playBtn').textContent = '▶';
}

function togglePlay(){
  if(playing){ stopPlay(); }
  else { startPlay(taskId_); }
}

function seekTo(val){
  curFrame = parseInt(val);
  document.getElementById('timeLbl').textContent = curFrame + ' / ' + frames;
  document.getElementById('preview').src = '/api/frame/' + taskId_ + '/' + curFrame + '?t=' + Date.now();
}

function showStats(s){
  if(!s) return;
  const panel = document.getElementById('statsPanel');
  panel.style.display = 'block';
  const items = [
    {v: s.frames,         l: '总帧数'},
    {v: s.total_ids,      l: '追踪 ID 总数'},
    {v: s.avg_det_ms+'ms',  l: '平均检测耗时'},
    {v: s.avg_track_ms+'ms',l: '平均追踪耗时'},
    {v: s.avg_fps+'fps',  l: '综合帧率'},
    {v: 'fp16',           l: '推理精度'},
  ];
  document.getElementById('statsGrid').innerHTML = items.map(i =>
    `<div class="stat-card"><div class="stat-val">${i.v}</div><div class="stat-lbl">${i.l}</div></div>`
  ).join('');
}

function showError(msg){
  const b = document.getElementById('errBox');
  b.textContent = '错误：' + msg; b.style.display = 'block';
  document.getElementById('progWrap').classList.remove('show');
}
</script>
</body>
</html>"""

# ── Flask 路由 ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/track', methods=['POST'])
def start_track():
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {'status': 'pending', 'progress': 0, 'result': None,
                      'error': None, 'stats': None, 'frame_count': 0, 'fps': 25}

    conf_thresh  = float(request.form.get('conf_thresh',  0.40))
    nms_thresh   = float(request.form.get('nms_thresh',   0.70))
    track_thresh = float(request.form.get('track_thresh', 0.70))

    if request.form.get('demo'):
        video_path = DEMO_VIDEO
    else:
        f = request.files.get('video')
        if not f:
            return jsonify({'error': '未收到视频'}), 400
        video_path = os.path.join(UPLOAD_DIR, f'{task_id}_{f.filename}')
        f.save(video_path)

    t = threading.Thread(target=run_tracking,
                         args=(task_id, video_path, conf_thresh, nms_thresh, track_thresh),
                         daemon=True)
    t.start()
    return jsonify({'task_id': task_id})

@app.route('/api/status/<task_id>')
def get_status(task_id):
    t = tasks.get(task_id)
    if not t:
        return jsonify({'error': 'not found'}), 404
    return jsonify(t)

@app.route('/api/frame/<task_id>/<int:frame_idx>')
def get_frame(task_id, frame_idx):
    """返回指定帧的 JPEG 图片"""
    frame_path = os.path.join(FRAMES_DIR, task_id, f'{frame_idx:06d}.jpg')
    if not os.path.exists(frame_path):
        return '', 404
    return send_file(frame_path, mimetype='image/jpeg')

@app.route('/api/result/<task_id>')
def get_result(task_id):
    """下载 mp4v 原始视频"""
    t = tasks.get(task_id)
    if not t or not t.get('result') or not os.path.exists(t['result']):
        return jsonify({'error': 'not ready'}), 404
    return send_file(
        t['result'],
        mimetype='video/mp4',
        as_attachment=True,
        download_name=f'bytetrack_{task_id}.mp4'
    )

if __name__ == '__main__':
    print('ByteTrack Web Service → http://0.0.0.0:5002')
    app.run(host='0.0.0.0', port=5002, threaded=True)
