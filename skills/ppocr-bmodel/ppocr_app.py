import os
import sys
import cv2
import numpy as np
import base64
import json
import time
import copy
import math
import argparse
import logging
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

logging.basicConfig(level=logging.INFO)

# Add PP-OCR python path
sys.path.insert(0, '/data/sophon-demo/sample/PP-OCR/python')
os.chdir('/data/sophon-demo/sample/PP-OCR/python')

import ppocr_det_opencv as predict_det
import ppocr_rec_opencv as predict_rec
import ppocr_cls_opencv as predict_cls

app = Flask(__name__)

# ---- OCR logic (from ppocr_system_opencv.py) ----

def get_rotate_crop_image(img, points):
    assert len(points) == 4
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    img_crop_width = max(16, img_crop_width)
    img_crop_height = max(16, img_crop_height)
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height),
                                  borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if dst_img.shape[0] * 1.0 / dst_img.shape[1] >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def sorted_boxes_dict(dt_boxes_dict):
    sorted_list = sorted(zip(*dt_boxes_dict.values()), key=lambda x: (x[0][0][1], x[0][0][0]))
    result = {}
    result["dt_boxes"], result["text"], result["score"] = map(list, zip(*sorted_list))
    return result

def draw_ocr_box_txt(image, boxes, txts, scores, rec_thresh=0.5,
                     font_path='/data/sophon-demo/sample/PP-OCR/datasets/fonts/simfang.ttf'):
    import random
    random.seed(0)
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))
    from PIL import ImageDraw, ImageFont
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < rec_thresh:
            continue
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon([box[0][0], box[0][1], box[1][0], box[1][1],
                            box[2][0], box[2][1], box[3][0], box[3][1]], outline=color)
        box_height = math.sqrt((box[0][0]-box[3][0])**2 + (box[0][1]-box[3][1])**2)
        box_width = math.sqrt((box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


class OCRArgs:
    dev_id = 0
    bmodel_det = '/data/sophon-demo/sample/PP-OCR/models/BM1684X/ch_PP-OCRv4_det_fp32.bmodel'
    det_limit_side_len = [640]
    bmodel_rec = '/data/sophon-demo/sample/PP-OCR/models/BM1684X/ch_PP-OCRv4_rec_fp32.bmodel'
    img_size = [[320, 48], [640, 48]]
    char_dict_path = '/data/sophon-demo/sample/PP-OCR/datasets/ppocr_keys_v1.txt'
    use_space_char = True
    use_beam_search = False
    beam_size = 5
    rec_thresh = 0.5
    use_angle_cls = False
    bmodel_cls = ''
    label_list = ['0', '180']
    cls_thresh = 0.9


# Load models once at startup
logging.info("Loading PP-OCR models...")
_args = OCRArgs()
text_detector = predict_det.PPOCRv2Det(_args)
text_recognizer = predict_rec.PPOCRv2Rec(_args)
logging.info("Models loaded.")


def run_ocr(img_bgr):
    img_list = [img_bgr]
    dt_boxes_list = text_detector(img_list)
    img_dict = {"imgs": [], "dt_boxes": [], "pic_ids": []}
    for id, dt_boxes in enumerate(dt_boxes_list):
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(img_list[id], tmp_box)
            img_dict["imgs"].append(img_crop)
            img_dict["dt_boxes"].append(dt_boxes[bno])
            img_dict["pic_ids"].append(id)
    rec_res = text_recognizer(img_dict["imgs"])
    results = {"dt_boxes": [], "text": [], "score": []}
    for i, id in enumerate(rec_res.get("ids")):
        text, score = rec_res["res"][i]
        if score >= _args.rec_thresh:
            results["dt_boxes"].append(img_dict["dt_boxes"][id])
            results["text"].append(text)
            results["score"].append(score)
    if results["dt_boxes"]:
        results = sorted_boxes_dict(results)
    return results


# ---- HTML page ----

HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PP-OCR 文字识别</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f0f2f5; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; padding: 20px 40px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
  .header h1 { font-size: 24px; font-weight: 600; }
  .header p { font-size: 13px; opacity: 0.85; margin-top: 4px; }
  .container { max-width: 1100px; margin: 30px auto; padding: 0 20px; }
  .card { background: white; border-radius: 12px; padding: 28px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 24px; }
  .upload-area { border: 2px dashed #1a73e8; border-radius: 10px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.3s; background: #f8fbff; }
  .upload-area:hover { background: #e8f0fe; border-color: #0d47a1; }
  .upload-area.dragging { background: #e8f0fe; border-color: #0d47a1; transform: scale(1.01); }
  .upload-icon { font-size: 48px; margin-bottom: 12px; }
  .upload-area p { color: #555; font-size: 15px; }
  .upload-area small { color: #999; }
  #fileInput { display: none; }
  .btn { display: inline-block; padding: 10px 28px; background: #1a73e8; color: white; border: none; border-radius: 8px; font-size: 15px; cursor: pointer; transition: background 0.2s; }
  .btn:hover { background: #0d47a1; }
  .btn:disabled { background: #aaa; cursor: not-allowed; }
  .preview-wrap { display: flex; gap: 20px; flex-wrap: wrap; }
  .preview-box { flex: 1; min-width: 280px; }
  .preview-box h3 { font-size: 14px; color: #666; margin-bottom: 8px; font-weight: 500; }
  .preview-box img { width: 100%; border-radius: 8px; border: 1px solid #eee; }
  .result-table { width: 100%; border-collapse: collapse; font-size: 14px; }
  .result-table th { background: #f1f3f4; padding: 10px 14px; text-align: left; color: #555; font-weight: 600; border-bottom: 2px solid #e0e0e0; }
  .result-table td { padding: 10px 14px; border-bottom: 1px solid #f0f0f0; color: #333; }
  .result-table tr:hover td { background: #f8fbff; }
  .score-bar { display: inline-block; height: 6px; background: #1a73e8; border-radius: 3px; vertical-align: middle; margin-right: 6px; }
  .loading { display: none; text-align: center; padding: 30px; color: #666; }
  .spinner { width: 40px; height: 40px; border: 4px solid #f0f0f0; border-top-color: #1a73e8; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 12px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error { background: #fff3f3; border: 1px solid #ffcdd2; color: #c62828; padding: 12px 16px; border-radius: 8px; }
  .stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
  .stat { background: #f1f8ff; border-radius: 8px; padding: 10px 18px; font-size: 13px; color: #1a73e8; }
  .stat strong { font-size: 20px; display: block; }
  #results { display: none; }
</style>
</head>
<body>
<div class="header">
  <h1>PP-OCR 文字识别系统</h1>
  <p>基于 Sophon BM1684X 加速推理 &nbsp;|&nbsp; PP-OCRv4 模型</p>
</div>
<div class="container">
  <div class="card">
    <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <div class="upload-icon">🖼️</div>
      <p>点击或拖拽图片到此处上传</p>
      <small>支持 JPG、PNG、BMP、WEBP 格式</small>
      <br><br>
      <button class="btn" onclick="event.stopPropagation();document.getElementById('fileInput').click()">选择图片</button>
    </div>
    <input type="file" id="fileInput" accept="image/*">
    <div style="margin-top:16px;text-align:center;">
      <button class="btn" id="runBtn" disabled onclick="runOCR()">开始识别</button>
    </div>
  </div>

  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p>正在识别中，请稍候...</p>
  </div>

  <div id="results">
    <div class="card">
      <div class="stats" id="stats"></div>
      <div class="preview-wrap" id="previewWrap"></div>
    </div>
    <div class="card">
      <h2 style="font-size:16px;margin-bottom:16px;color:#333">识别结果</h2>
      <div id="tableWrap"></div>
    </div>
  </div>
</div>

<script>
let selectedFile = null;

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const runBtn = document.getElementById('runBtn');

fileInput.onchange = e => selectFile(e.target.files[0]);

dropZone.ondragover = e => { e.preventDefault(); dropZone.classList.add('dragging'); };
dropZone.ondragleave = () => dropZone.classList.remove('dragging');
dropZone.ondrop = e => { e.preventDefault(); dropZone.classList.remove('dragging'); selectFile(e.dataTransfer.files[0]); };

function selectFile(file) {
  if (!file) return;
  selectedFile = file;
  runBtn.disabled = false;
  document.getElementById('results').style.display = 'none';
  // show preview
  const reader = new FileReader();
  reader.onload = e => {
    dropZone.innerHTML = `<img src="${e.target.result}" style="max-height:200px;border-radius:8px;"><p style="margin-top:8px;color:#555">${file.name}</p>`;
  };
  reader.readAsDataURL(file);
}

async function runOCR() {
  if (!selectedFile) return;
  runBtn.disabled = true;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('results').style.display = 'none';

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const resp = await fetch('/ocr', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    showResults(data);
  } catch(e) {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('results').style.display = 'block';
    document.getElementById('results').innerHTML = `<div class="card"><div class="error">错误：${e.message}</div></div>`;
  } finally {
    runBtn.disabled = false;
    document.getElementById('loading').style.display = 'none';
  }
}

function showResults(data) {
  const resultsDiv = document.getElementById('results');
  resultsDiv.style.display = 'block';

  // stats
  document.getElementById('stats').innerHTML = `
    <div class="stat"><strong>${data.count}</strong>识别到的文字块</div>
    <div class="stat"><strong>${data.time_ms} ms</strong>推理耗时</div>
  `;

  // images
  document.getElementById('previewWrap').innerHTML = `
    <div class="preview-box"><h3>原图</h3><img src="data:image/jpeg;base64,${data.original_b64}"></div>
    <div class="preview-box"><h3>识别结果可视化</h3><img src="data:image/jpeg;base64,${data.result_b64}"></div>
  `;

  // table
  let rows = data.texts.map((t, i) => `
    <tr>
      <td>${i+1}</td>
      <td>${t}</td>
      <td>
        <span class="score-bar" style="width:${Math.round(data.scores[i]*60)}px"></span>
        ${(data.scores[i]*100).toFixed(1)}%
      </td>
    </tr>
  `).join('');

  document.getElementById('tableWrap').innerHTML = rows ? `
    <table class="result-table">
      <thead><tr><th>#</th><th>识别文字</th><th>置信度</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  ` : '<p style="color:#999;text-align:center;padding:20px">未识别到文字</p>';
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': '未上传图片'}), 400
    file = request.files['image']
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({'error': '图片解码失败'}), 400

    t0 = time.time()
    results = run_ocr(img_bgr)
    elapsed_ms = int((time.time() - t0) * 1000)

    # Draw result image
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if results["dt_boxes"]:
        vis = draw_ocr_box_txt(pil_img, results["dt_boxes"], results["text"], results["score"])
    else:
        vis = np.array(pil_img)

    def to_b64(arr):
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format='JPEG', quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    orig_arr = np.array(pil_img)
    return jsonify({
        'count': len(results["text"]),
        'texts': results["text"],
        'scores': [float(s) for s in results["score"]],
        'time_ms': elapsed_ms,
        'original_b64': to_b64(orig_arr),
        'result_b64': to_b64(vis),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8899, debug=False)
