# OpenCV and Flask modules
import time
import numpy
import cv2
from flask import Flask, Response, render_template, request
from PIL import Image

# Facial Recognition
import face_recognition
from simple_facerec import SimpleFacerec
# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


# Flask App
app = Flask(__name__)
# setup camera and resolution
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

@app.route("/verify", methods = ['GET', 'POST'])
def verify():
    if request.method == 'GET':
        return render_template('upload.html')
    else:
        image = cv2.imdecode(numpy.fromstring(request.files['file'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        face_locations, face_names = sfr.detect_known_faces(image)
        name = request.form['name']
        for face_loc, face_name in zip(face_locations, face_names):
            if name == face_name:
                return Response("Yes")
                break
        else:
            return Response("No")