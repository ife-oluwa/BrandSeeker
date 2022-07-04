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
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync


UPLOAD_FOLDER = './images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'pdf', 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.from_object('config')
DETECTION_URL = "/v1/object-detection/yolov5s"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

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

@app.route('/results')
def results():
    return render_template('results.html')

@app.route(DETECTION_URL, methods=["POST", "GET"])
def predict():
    
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=True)
            source = UPLOAD_FOLDER
            # stride, names, pt = model.stride, model.names, model.pt
            # imgsz = check_img_size((640, 640), s=stride)  # check image size might want to remove
            # device = select_device('')
            # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            # bs = 1  # batch_size

            # dt, seen = [0.0, 0.0, 0.0], 0
            # for path, im, im0s, vid_cap, s in dataset:
            #     im = torch.from_numpy(im).to(device)
            #     im = im.float()
            #     im /= 255  # 0 - 255 to 0.0 - 1.0

            #     if len(im.shape) == 3:
            #         im = im[None]  # expand for batch dim
            #     t2 = time_sync()


            #     # Inference
            #     pred = model(im)
            #     t3 = time_sync()


            #     # NMS
            #     pred = non_max_suppression(pred, conf_thres=0.35, max_det=5) # might want to discuss the max nb of detection, iou...


            #     print(pred)
            pred_image = model('./images/uber-eats-1024x682.jpg')
            image = pred_image.print()
            return render_template('results.html', image=pred_image)
    return render_template('predict.html')


if __name__ == "__main__":
    app.run(debug=True) 
