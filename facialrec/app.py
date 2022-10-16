# OpenCV and Flask modules
import os
import time
import numpy
import cv2
import pathlib
from flask import Flask, Response, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
from PIL import Image

# Facial Recognition
import face_recognition
from simple_facerec import SimpleFacerec

from pose_rec import PostureRec

# Flask App
app = Flask(__name__)
# setup camera and resolution
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

app.config['UPLOAD_FOLDER'] = "./static/images"
app.config['SECRET_KEY'] = "randomsecret"

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images(app.config['UPLOAD_FOLDER'])
pr = PostureRec()

# Function to encode facial recognition images on cv2 capture
def facial_recognition():
    while True:
        time.sleep(0.1)
        ret, frame = cam.read()
        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        _, img = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: imageviewer/jpeg\r\n\r\n' + img.tobytes() + b'\r\n')

# Routing to Face Recognition
# To change to capture images from video stream, change facial_recognition_image() function to facial_recognition()
@app.route("/")
def facial_rec():
    return Response(facial_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

# GET: POC to verify
# POST: Endpoint
@app.route("/verify", methods = ['GET', 'POST'])
def verify():
    if request.method == 'GET':
        return render_template('verify.html')
    else:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        image = cv2.imdecode(numpy.fromstring(file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        face_locations, face_names = sfr.detect_known_faces(image)
        name = request.form['name']
        for face_loc, face_name in zip(face_locations, face_names):
            if name == face_name:
                response = Response("MATCH\nExpected {}\nGot {}\n".format(name, (" ".join(face_names))))
                response.headers["content-type"] = "text/plain"
                return response
        response = Response("FAIL\nExpected {}\nGot {}\n".format(name, (" ".join(face_names))))
        response.headers["content-type"] = "text/plain"
        return response

def generate_filename(username, original_file_name):
    return username + pathlib.Path(original_file_name).suffix

@app.route('/upload', methods = ['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    else:
        name = request.form['name']
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        filename = generate_filename(name, file.filename)
        pathname = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pathname)
        sfr.load_encoding_images("./static/images/")
        response = Response("Uploaded to {}!\n".format(pathname))
        response.headers["content-type"] = "text/plain"
        return response

@app.route('/detect', methods = ['GET', 'POST'])
def detect():
    if request.method == 'GET':
        return render_template('detect.html')
    else:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        response = Response(pr.detect_posture(file))
        response.headers["content-type"] = "text/plain"
        return response

@app.route("/list")
def list():
    pics = os.listdir(app.config['UPLOAD_FOLDER'])
    pics = filter(lambda x: not x.startswith("."), pics)
    abspath = app.config['UPLOAD_FOLDER']
    pics = [
        (   pathlib.Path(os.path.basename(pic)).stem,
            os.path.join(abspath, pic)
        )
        for pic in pics
    ]
    return render_template('list.html', pics = pics)