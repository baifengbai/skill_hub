import os
import sys
import cv2
import numpy as np
import sophon.sail as sail
import base64
import time
from flask import Flask, request, jsonify, render_template_string, Response
from urllib.parse import quote, unquote

# 确保能 import YOLOv5 依赖
sys.path.insert(0, '/data/sophon-demo/sample/YOLOv5/python')
from postprocess_numpy import PostProcess
from utils import COLORS, COCO_CLASSES

app = Flask(__name__)
os.makedirs('/tmp/up', exist_ok=True)

BMODEL = '/data/sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel'
IMAGE_DIR = '/data/sophon-demo/sample/ResNet/datasets/imagenet_val_1k/img'

# ── 模型封装 ──────────────────────────────────────────────────────────────────

class YOLOv5:
    def __init__(self, bmodel, dev_id=0, conf_thresh=0.25, nms_thresh=0.45):
        self.net = sail.Engine(bmodel, dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.postprocess = PostProcess(
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
            agnostic=False,
            multi_label=True,
            max_det=1000,
        )

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        shape = im.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, (r, r), (dw, dh)

    def preprocess(self, img):
        lb, ratio, (tx1, ty1) = self.letterbox(img, (self.net_h, self.net_w))
        lb = lb.transpose((2, 0, 1))[::-1].astype(np.float32)
        lb = np.ascontiguousarray(lb / 255.0)
        return lb, ratio, (tx1, ty1)

    def detect(self, img):
        """img: BGR numpy array, 返回 list of {label, conf, box:[x1,y1,x2,y2]}"""
        ori_h, ori_w = img.shape[:2]
        pre, ratio, txy = self.preprocess(img)
        inp = np.expand_dims(pre, 0)
        if self.batch_size > 1:
            pad = np.zeros(self.input_shape, dtype=np.float32)
            pad[0] = inp[0]
            inp = pad

        outputs = self.net.process(self.graph_name, {self.input_name: inp})
        out_keys = list(outputs.keys())
        ord_ = [out_keys.index(n) for n in self.output_names if n in out_keys]
        out = [outputs[out_keys[i]][:1] for i in ord_]

        results = self.postprocess(out, [(ori_w, ori_h)], [ratio], [txy])
        det = results[0]  # shape (N, 6): x1,y1,x2,y2,score,class_id

        detections = []
        for row in det:
            x1, y1, x2, y2, score, cls_id = row
            cls_id = int(cls_id)
            label = COCO_CLASSES[cls_id + 1] if cls_id + 1 < len(COCO_CLASSES) else str(cls_id)
            detections.append({
                'label': label,
                'conf': float(round(score, 4)),
                'box': [float(round(x1)), float(round(y1)), float(round(x2)), float(round(y2))]
            })
        return detections

    def draw(self, img, detections):
        img = img.copy()
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d['box']]
            cls_id = list(COCO_CLASSES).index(d['label']) - 1 if d['label'] in COCO_CLASSES else 0
            color = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{d['label']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img


model = YOLOv5(BMODEL)

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv5 目标检测</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
            min-height: 100vh; padding: 20px; color: #e6edf3;
        }
        .container {
            max-width: 1100px; margin: 0 auto;
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(12px);
            border-radius: 24px; padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .header { text-align: center; margin-bottom: 28px; }
        .header h1 {
            font-size: 2rem;
            background: linear-gradient(90deg, #58a6ff, #3fb950);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin-bottom: 8px;
        }
        .header a { color: #58a6ff; text-decoration: none; font-size: 0.9rem; }
        .header a:hover { text-decoration: underline; }
        .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        @media (max-width: 768px) { .main-content { grid-template-columns: 1fr; } }
        .panel {
            background: rgba(0,0,0,0.25);
            border-radius: 16px; padding: 20px;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .panel-title {
            font-size: 0.95rem; color: #58a6ff;
            margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
        }
        .upload-area {
            border: 2px dashed rgba(88,166,255,0.3);
            border-radius: 12px; padding: 28px;
            text-align: center; cursor: pointer; transition: all 0.3s;
            margin-bottom: 14px;
        }
        .upload-area:hover { border-color: #58a6ff; background: rgba(88,166,255,0.05); }
        .upload-area.dragover { border-color: #3fb950; background: rgba(63,185,80,0.08); }
        .upload-icon { font-size: 36px; margin-bottom: 8px; }
        .upload-text { color: #8b949e; font-size: 0.85rem; }
        #file-input { display: none; }
        .gallery {
            display: grid; grid-template-columns: repeat(5, 1fr);
            gap: 6px; max-height: 180px; overflow-y: auto; margin-bottom: 14px;
        }
        .gallery img {
            width: 100%; aspect-ratio: 1; object-fit: cover;
            border-radius: 6px; cursor: pointer; transition: all 0.2s;
            border: 2px solid transparent;
        }
        .gallery img:hover { transform: scale(1.05); }
        .gallery img.selected { border-color: #58a6ff; }
        .preview-wrap {
            position: relative; background: rgba(0,0,0,0.3);
            border-radius: 12px; overflow: hidden;
            min-height: 200px; display: flex; align-items: center; justify-content: center;
        }
        #preview { width: 100%; max-height: 340px; object-fit: contain; display: none; }
        .preview-placeholder { color: #484f58; font-size: 0.85rem; }
        .param-row { display: flex; gap: 12px; margin-bottom: 14px; }
        .param-group { flex: 1; }
        .param-group label { display: block; color: #8b949e; font-size: 0.8rem; margin-bottom: 6px; }
        .param-group input[type=range] { width: 100%; accent-color: #58a6ff; }
        .param-group span { color: #58a6ff; font-size: 0.85rem; font-weight: 600; }
        .btn {
            width: 100%; padding: 13px; font-size: 1rem; font-weight: 600;
            background: linear-gradient(90deg, #238636, #2ea043);
            border: none; border-radius: 12px; color: #fff;
            cursor: pointer; transition: all 0.3s;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(46,160,67,0.4); }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
        .loading { display: none; text-align: center; padding: 30px; }
        .loading.show { display: block; }
        .spinner {
            width: 44px; height: 44px;
            border: 3px solid rgba(88,166,255,0.2);
            border-top-color: #58a6ff;
            border-radius: 50%; animation: spin 0.9s linear infinite; margin: 0 auto 12px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .det-list { margin-top: 14px; max-height: 340px; overflow-y: auto; }
        .det-item {
            display: flex; justify-content: space-between; align-items: center;
            background: rgba(255,255,255,0.04); border-radius: 10px;
            padding: 10px 14px; margin-bottom: 8px;
            border-left: 3px solid #58a6ff;
            transition: all 0.2s;
        }
        .det-item:hover { background: rgba(88,166,255,0.08); }
        .det-label { font-weight: 600; font-size: 0.95rem; }
        .det-conf { color: #3fb950; font-weight: 700; font-size: 0.9rem; }
        .det-box { color: #8b949e; font-size: 0.75rem; margin-top: 2px; }
        .empty-state { text-align: center; padding: 30px; color: #484f58; }
        .stats-bar {
            display: flex; gap: 16px; margin-bottom: 14px; flex-wrap: wrap;
        }
        .stat-chip {
            background: rgba(88,166,255,0.1); border: 1px solid rgba(88,166,255,0.2);
            padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; color: #58a6ff;
        }
        .result-img-wrap { position: relative; margin-bottom: 14px; }
        #result-img { width: 100%; border-radius: 12px; display: none; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>YOLOv5 目标检测</h1>
        <a href="/api-doc" target="_blank">📚 API 文档</a>
    </div>
    <div class="main-content">
        <!-- 左：输入 -->
        <div class="panel">
            <div class="panel-title">🖼️ 选择图片</div>
            <div class="upload-area" id="upload-area">
                <div class="upload-icon">📁</div>
                <div class="upload-text">点击或拖拽图片到这里</div>
                <input type="file" id="file-input" accept="image/*">
            </div>
            <div class="gallery" id="gallery"></div>
            <div class="preview-wrap">
                <img id="preview">
                <div class="preview-placeholder" id="preview-placeholder">图片预览</div>
            </div>
        </div>

        <!-- 右：参数 + 结果 -->
        <div class="panel">
            <div class="panel-title">⚙️ 参数设置</div>
            <div class="param-row">
                <div class="param-group">
                    <label>置信度阈值：<span id="conf-val">0.25</span></label>
                    <input type="range" id="conf" min="0.05" max="0.95" step="0.05" value="0.25"
                           oninput="document.getElementById('conf-val').textContent=parseFloat(this.value).toFixed(2)">
                </div>
                <div class="param-group">
                    <label>NMS 阈值：<span id="nms-val">0.45</span></label>
                    <input type="range" id="nms" min="0.1" max="0.9" step="0.05" value="0.45"
                           oninput="document.getElementById('nms-val').textContent=parseFloat(this.value).toFixed(2)">
                </div>
            </div>
            <button class="btn" id="detect-btn" onclick="doDetect()">开始检测</button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div>检测中...</div>
            </div>

            <div id="result-section">
                <div class="result-img-wrap">
                    <img id="result-img">
                </div>
                <div class="stats-bar" id="stats-bar"></div>
                <div class="det-list" id="det-list">
                    <div class="empty-state">选择图片后点击检测</div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    let selectedFile = null;
    let selectedPath = null;

    loadGallery();
    setupUpload();

    function setupUpload() {
        const area = document.getElementById('upload-area');
        const input = document.getElementById('file-input');
        area.onclick = () => input.click();
        area.ondragover = e => { e.preventDefault(); area.classList.add('dragover'); };
        area.ondragleave = () => area.classList.remove('dragover');
        area.ondrop = e => {
            e.preventDefault(); area.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        };
        input.onchange = e => { if (e.target.files.length) handleFile(e.target.files[0]); };
    }

    function handleFile(file) {
        selectedFile = file; selectedPath = null;
        const reader = new FileReader();
        reader.onload = e => showPreview(e.target.result);
        reader.readAsDataURL(file);
        clearGallerySelection();
    }

    function showPreview(src) {
        const p = document.getElementById('preview');
        p.src = src; p.style.display = 'block';
        document.getElementById('preview-placeholder').style.display = 'none';
        document.getElementById('result-img').style.display = 'none';
    }

    async function loadGallery() {
        try {
            const res = await fetch('/api/images');
            const images = await res.json();
            const g = document.getElementById('gallery');
            g.innerHTML = '';
            images.forEach(img => {
                const i = document.createElement('img');
                i.src = img.thumb; i.title = img.name;
                i.onclick = () => {
                    clearGallerySelection(); i.classList.add('selected');
                    selectedFile = null; selectedPath = img.path;
                    showPreview(`/api/preview?p=${encodeURIComponent(img.path)}`);
                };
                g.appendChild(i);
            });
        } catch(e) { console.error(e); }
    }

    function clearGallerySelection() {
        document.querySelectorAll('.gallery img').forEach(x => x.classList.remove('selected'));
    }

    async function doDetect() {
        if (!selectedFile && !selectedPath) { alert('请先选择图片'); return; }
        const conf = document.getElementById('conf').value;
        const nms = document.getElementById('nms').value;
        const btn = document.getElementById('detect-btn');
        const loading = document.getElementById('loading');

        btn.disabled = true; loading.classList.add('show');
        document.getElementById('det-list').innerHTML = '';
        document.getElementById('stats-bar').innerHTML = '';

        try {
            const fd = new FormData();
            if (selectedFile) fd.append('file', selectedFile);
            else fd.append('path', selectedPath);
            fd.append('conf_thresh', conf);
            fd.append('nms_thresh', nms);

            const res = await fetch('/api/detect', { method: 'POST', body: fd });
            const r = await res.json();
            if (r.error) { alert(r.error); return; }

            // 结果图
            const ri = document.getElementById('result-img');
            ri.src = r.image; ri.style.display = 'block';
            document.getElementById('preview').style.display = 'none';

            // 统计栏
            const stats = document.getElementById('stats-bar');
            const classCounts = {};
            r.detections.forEach(d => { classCounts[d.label] = (classCounts[d.label] || 0) + 1; });
            stats.innerHTML = `<span class="stat-chip">共 ${r.detections.length} 个目标</span>` +
                `<span class="stat-chip">耗时 ${r.time_ms}ms</span>` +
                Object.entries(classCounts).map(([k,v]) =>
                    `<span class="stat-chip">${k}: ${v}</span>`).join('');

            // 列表
            const list = document.getElementById('det-list');
            if (r.detections.length === 0) {
                list.innerHTML = '<div class="empty-state">未检测到目标</div>';
            } else {
                list.innerHTML = r.detections.map(d => `
                    <div class="det-item">
                        <div>
                            <div class="det-label">${d.label}</div>
                            <div class="det-box">[${d.box.map(v=>Math.round(v)).join(', ')}]</div>
                        </div>
                        <div class="det-conf">${(d.conf*100).toFixed(1)}%</div>
                    </div>`).join('');
            }
        } catch(e) {
            alert('检测失败: ' + e.message);
        } finally {
            btn.disabled = false; loading.classList.remove('show');
        }
    }
</script>
</body>
</html>"""

# ── API 文档 ──────────────────────────────────────────────────────────────────

API_DOC = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>YOLOv5 API 文档</title>
    <style>
        body { font-family: 'Monaco','Menlo',monospace; background:#1e1e1e; color:#d4d4d4; padding:20px; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color:#569cd6; border-bottom:1px solid #333; padding-bottom:10px; }
        h2 { color:#4ec9b0; margin-top:30px; }
        h3 { color:#dcdcaa; margin-top:20px; }
        .endpoint { background:#2d2d2d; padding:15px; border-radius:8px; margin:15px 0; border-left:4px solid #569cd6; }
        .method { display:inline-block; padding:4px 10px; border-radius:4px; font-weight:bold; margin-right:10px; }
        .get  { background:#61affe; color:#fff; }
        .post { background:#49cc90; color:#fff; }
        .url  { color:#9cdcfe; }
        .desc { color:#808080; margin-top:8px; }
        code { background:#3c3c3c; padding:2px 6px; border-radius:3px; color:#ce9178; }
        pre  { background:#2d2d2d; padding:15px; border-radius:8px; overflow-x:auto; line-height:1.5; }
        table { width:100%; border-collapse:collapse; margin:10px 0; }
        th,td { text-align:left; padding:8px; border-bottom:1px solid #333; }
        th { color:#569cd6; }
        .back { position:fixed; top:20px; right:20px; background:#569cd6; color:#fff;
                padding:10px 20px; text-decoration:none; border-radius:5px; }
    </style>
</head>
<body>
<a href="/" class="back">← 返回首页</a>
<div class="container">
    <h1>YOLOv5 目标检测 API 文档</h1>

    <h2>1. 目标检测接口</h2>
    <div class="endpoint">
        <span class="method post">POST</span>
        <span class="url">/api/detect</span>
        <p class="desc">上传图片，返回检测框、类别、置信度及带标注的结果图</p>
    </div>

    <h3>图片输入（三选一）</h3>
    <table>
        <tr><th>字段</th><th>类型</th><th>说明</th></tr>
        <tr><td><code>file</code></td><td>file</td><td>multipart/form-data 上传图片文件</td></tr>
        <tr><td><code>path</code></td><td>string</td><td>服务器上图片的绝对路径</td></tr>
        <tr><td><code>image_base64</code></td><td>string</td><td>Base64 编码图片（可带 data:image 前缀）</td></tr>
    </table>

    <h3>可选参数</h3>
    <table>
        <tr><th>字段</th><th>类型</th><th>默认值</th><th>说明</th></tr>
        <tr><td><code>conf_thresh</code></td><td>float</td><td>0.25</td><td>置信度阈值 (0~1)</td></tr>
        <tr><td><code>nms_thresh</code></td><td>float</td><td>0.45</td><td>NMS IOU 阈值 (0~1)</td></tr>
    </table>

    <h3>响应格式</h3>
    <pre><code>{
  "detections": [
    {
      "label": "person",
      "conf": 0.872,
      "box": [120.0, 45.0, 380.0, 610.0]   // [x1, y1, x2, y2]
    }
  ],
  "count": 1,
  "time_ms": 58,
  "image": "data:image/jpeg;base64,..."     // 带标注框的结果图
}</code></pre>

    <h3>cURL 示例 — 文件上传</h3>
    <pre><code>curl -X POST http://<DEVICE_IP>:15001/api/detect \\
  -F "file=@/path/to/image.jpg" \\
  -F "conf_thresh=0.25" \\
  -F "nms_thresh=0.45"</code></pre>

    <h3>cURL 示例 — Base64</h3>
    <pre><code>IMG_B64=$(base64 -w0 image.jpg)
curl -X POST http://<DEVICE_IP>:15001/api/detect \\
  -F "image_base64=${IMG_B64}" \\
  -F "conf_thresh=0.3"</code></pre>

    <h3>Python 示例 — 文件上传</h3>
    <pre><code>import requests

resp = requests.post(
    "http://<DEVICE_IP>:15001/api/detect",
    files={"file": open("image.jpg", "rb")},
    data={"conf_thresh": 0.25, "nms_thresh": 0.45}
)
data = resp.json()
print(f"检测到 {data['count']} 个目标，耗时 {data['time_ms']}ms")
for d in data["detections"]:
    print(f"  {d['label']}  conf={d['conf']:.3f}  box={d['box']}")</code></pre>

    <h3>Python 示例 — Base64</h3>
    <pre><code>import requests, base64

with open("image.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://<DEVICE_IP>:15001/api/detect",
    data={"image_base64": b64, "conf_thresh": 0.25}
)
print(resp.json())</code></pre>

    <h3>Python 示例 — 服务器路径</h3>
    <pre><code>import requests

resp = requests.post(
    "http://<DEVICE_IP>:15001/api/detect",
    data={
        "path": "/data/sophon-demo/sample/ResNet/datasets/imagenet_val_1k/img/ILSVRC2012_val_00000075.JPEG",
        "conf_thresh": 0.25
    }
)
print(resp.json())</code></pre>

    <h2>2. 图片列表接口</h2>
    <div class="endpoint">
        <span class="method get">GET</span>
        <span class="url">/api/images</span>
        <p class="desc">获取服务器测试图片列表（含缩略图 URL）</p>
    </div>
    <pre><code>curl http://<DEVICE_IP>:15001/api/images</code></pre>
    <pre><code>[
  {"name": "ILSVRC2012...", "path": "/data/.../xxx.JPEG", "thumb": "/api/thumb?p=..."},
  ...
]</code></pre>

    <h2>3. 缩略图接口</h2>
    <div class="endpoint">
        <span class="method get">GET</span>
        <span class="url">/api/thumb?p={图片绝对路径}</span>
        <p class="desc">返回 120px 宽的缩略图（JPEG 二进制）</p>
    </div>

    <h2>4. 预览图接口</h2>
    <div class="endpoint">
        <span class="method get">GET</span>
        <span class="url">/api/preview?p={图片绝对路径}</span>
        <p class="desc">返回最大边 800px 的预览图（JPEG 二进制）</p>
    </div>
</div>
</body>
</html>"""

# ── Flask 路由 ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api-doc')
def api_doc():
    return render_template_string(API_DOC)

@app.route('/api/images')
def list_images():
    images = []
    if os.path.exists(IMAGE_DIR):
        for f in sorted(os.listdir(IMAGE_DIR))[:30]:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                p = os.path.join(IMAGE_DIR, f)
                images.append({'name': f[:18], 'path': p,
                                'thumb': f'/api/thumb?p={quote(p, safe="")}'})
    return jsonify(images)

@app.route('/api/thumb')
def get_thumb():
    p = unquote(request.args.get('p', ''))
    if os.path.exists(p):
        img = cv2.imread(p)
        if img is not None:
            h, w = img.shape[:2]
            img = cv2.resize(img, (120, int(120 * h / w)))
            _, b = cv2.imencode('.jpg', img)
            return Response(b.tobytes(), mimetype='image/jpeg')
    return '', 404

@app.route('/api/preview')
def get_preview():
    p = unquote(request.args.get('p', ''))
    if os.path.exists(p):
        img = cv2.imread(p)
        if img is not None:
            h, w = img.shape[:2]
            ms = 800
            if max(h, w) > ms:
                s = ms / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            _, b = cv2.imencode('.jpg', img)
            return Response(b.tobytes(), mimetype='image/jpeg')
    return '', 404

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        conf_thresh = float(request.form.get('conf_thresh', 0.25))
        nms_thresh  = float(request.form.get('nms_thresh',  0.45))

        img_path = None
        if 'file' in request.files and request.files['file'].filename:
            f = request.files['file']
            p = os.path.join('/tmp/up', f.filename)
            f.save(p)
            img_path = p
        elif 'path' in request.form:
            img_path = unquote(request.form['path'])
        elif 'image_base64' in request.form:
            raw = request.form['image_base64']
            if ',' in raw:
                raw = raw.split(',')[1]
            arr = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
            img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_cv is not None:
                img_path = f'/tmp/up/b64_{int(time.time())}.jpg'
                cv2.imwrite(img_path, img_cv)

        if not img_path or not os.path.exists(img_path):
            return jsonify({'error': '图片不存在'}), 400

        img = cv2.imread(img_path)
        if img is None:
            return jsonify({'error': '无法读取图片'}), 400

        # 动态阈值：重建模型 postprocess（轻量，不重新加载 bmodel）
        model.postprocess = PostProcess(
            conf_thresh=conf_thresh, nms_thresh=nms_thresh,
            agnostic=False, multi_label=True, max_det=1000)

        t0 = time.time()
        dets = model.detect(img)
        elapsed = int((time.time() - t0) * 1000)

        result_img = model.draw(img, dets)

        # 缩到最大边 900
        h, w = result_img.shape[:2]
        ms = 900
        if max(h, w) > ms:
            s = ms / max(h, w)
            result_img = cv2.resize(result_img, (int(w * s), int(h * s)))

        _, buf = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"

        return jsonify({
            'detections': dets,
            'count': len(dets),
            'time_ms': elapsed,
            'image': img_b64
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    print('=' * 50)
    print('YOLOv5 Web Service')
    print('Access: http://<DEVICE_IP>:15001')
    print('=' * 50)
    app.run(host='0.0.0.0', port=15001)
