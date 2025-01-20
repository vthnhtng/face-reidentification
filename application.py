import cv2
import time
from flask import Flask, render_template, request, Response, jsonify
from threading import Thread
from queue import Queue

from utils.config_loader import ConfigLoader
from modules.detectors.yolo import Yolo
from modules.recognizers.arcface import Arcface
from modules.embedding_storage.faiss import Faiss
from modules.face_reidentifier import FaceReidentifier

config_loader = ConfigLoader()

detector = Yolo(config_loader.get_yolo_weight())
recognizer = Arcface(config_loader.get_arcface_weight())
embedding_storage = Faiss(faiss_index=config_loader.get_faiss_index())

face_reidentifier = FaceReidentifier(
    detector=detector,
    recognizer=recognizer,
    embedding_storage=embedding_storage
)
print("Face reidentifier initialized.")

app = Flask(__name__)
frame_queue = Queue(maxsize=10)
fps = 15
frame_delay = 1.0 / fps
print("Flask app initialized.")

# define functions
def read_frames(camera):
    """Function to read frames from the video stream and push them into the frame queue."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

def gen_frames():
    """Function to encode frames and send them as part of the video stream."""
    camera = cv2.VideoCapture('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4')
    
    thread = Thread(target=read_frames, args=(camera,))
    thread.daemon = True
    thread.start()
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50]) # 50% quantity from original frame 

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            time.sleep(frame_delay)

    camera.release()

# Define the Flask routes
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/upload-image/<user_name>', methods=['POST'])
def upload_image(user_name):

    if 'image' in request.json:


        return jsonify({"message": f"Image for {user_name} received and saved successfully!"})
    else:
        return jsonify({"message": "No image data received"}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
