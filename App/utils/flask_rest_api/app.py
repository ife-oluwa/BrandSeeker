# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing a YOLOv5s model
"""

import argparse
import io
import os

import torch
from flask import Flask, request, redirect, url_for, flash, render_template
from PIL import Image
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'C:/videos/Users/videos/ifeol/videos/Documents/videos/Projects/videos/Computer vision/videos/BrandSeeker/videos/App/videos/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'pdf', 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.from_object('config')
DETECTION_URL = "/v1/object-detection/yolov5s"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/upload', methods=['POST', "GET"])
def upload():
    if request.method == 'POST':
        #print(request.files)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'file uploaded successfully'

@app.route(DETECTION_URL, methods=["POST", "GET"])
def predict():
    if request.method != "POST":
        return "HELLO WORLD"

    if request.files.get("image"):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        results = model(im, size=640)  # reduce size=320 for faster inference
        return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    opt = parser.parse_args()

    # Fix known issue urllib.error.HTTPError 403: rate limit exceeded https://github.com/ultralytics/yolov5/pull/7210
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # force_reload to recache
    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
