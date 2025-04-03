from detector.yolo import Yolo
from recognizer.arcface import ArcFace
from vector_storage.qdrant import Qdrant
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from model_service import ModelService
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from flask import Response
import time
import os
from PIL import Image
import cv2
from dotenv import load_dotenv

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Detector env
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS")
CONFIDENCE_THRESHOULD = float(os.getenv("CONFIDENCE_THRESHOLD"))

# Recognizer env
ARCFACE_WEIGHTS = os.getenv("ARCFACE_WEIGHTS")

# Vector database env
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME=os.getenv("COLLECTION_NAME")
SIMILARITY_THRESHOLD=float(os.getenv("SIMILARITY_THRESHOLD"))

yolo = Yolo(YOLO_WEIGHTS, CONFIDENCE_THRESHOULD)
arcface = ArcFace(ARCFACE_WEIGHTS)
qdrant = Qdrant(QDRANT_URL, COLLECTION_NAME, SIMILARITY_THRESHOLD)

model_service = ModelService(
    detector=yolo,
    recognizer=arcface,
    vector_storage=qdrant
)

users = {
    'admin': 'admin123',
}

def generate_frames():
    camera = cv2.VideoCapture(0)
    fps_limit = 10  # Limit to 10 frames per second
    frame_interval = 1 / fps_limit
    last_frame_time = time.time()

    while True:
        success, frame = camera.read()
        origin_frame = frame

        if not success:
            break

        current_time = time.time()
        if current_time - last_frame_time < frame_interval:
            continue  # Skip frames to maintain the FPS limit

        last_frame_time = current_time

        try:
            processed_frame = model_service.frame_process(frame)
        except Exception as e:
            processed_frame = origin_frame

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')

    camera.release()

def isLoggedIn():
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if isLoggedIn():
        return redirect(url_for('admin'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('admin'))
        else:
            flash('Invalid username or password', 'danger')
            return render_template('login.html')
    
    # Render login page for GET request
    return render_template('login.html')

@app.route('/admin')
def admin():
    if not isLoggedIn():
        flash('You need to login to access this page', 'warning')
        return redirect(url_for('login'))

    return render_template('admin.html')

@app.route('/dashboard')
def dashboard():
    # Get all image filenames in the 'static/images' folder
    image_folder = os.path.join(app.static_folder, 'logs')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    images = os.listdir(image_folder)

    # Filter out non-image files if necessary (optional)
    images = [image for image in images if image.endswith(('png', 'jpg', 'jpeg', 'gif'))]

    return render_template('dashboard.html', images=images)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if not isLoggedIn():
        return jsonify({"message": f"You need login first"})

    data = request.json
    if 'image' in data and 'username' in data:
        # Extract the base64-encoded image data
        image_data = data['image'].split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        username = data['username']

        user_folder = f"./faces/{username}/"
        os.makedirs(user_folder, exist_ok=True)
        existing_files = [f for f in os.listdir(user_folder) if f.endswith(".png")]
        next_index = len(existing_files) + 1
        file_path = os.path.join(user_folder, f"{next_index}.png")
    
        image.save(file_path)

        model_service.add_new_user(
            image = cv2.imread(file_path),
            user_data={
                'username': username,
                'image': file_path
            }
        )
    
        return jsonify({"message": f"Image for {username} received and saved successfully!"})
    else:
        return jsonify({"message": "No image data received"}), 400


if __name__ == '__main__':
    app.run()
