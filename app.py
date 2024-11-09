from flask import Flask, render_template, request, jsonify
from model_services import ModelServices
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from flask import Response
from uart_communication import UARTCommunicator

app = Flask(__name__)

model_services = ModelServices()
uart_communicator = UARTCommunicator()

def gen_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        processed_frame, face_detected = model_services.frame_process(frame)

        # Static variable to store the last time we sent a message
        if not hasattr(gen_frames, 'last_send_time'):
            gen_frames.last_send_time = 0

        # Get current time
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        # Check if 2 seconds have passed since last send
        if current_time - gen_frames.last_send_time >= 2.0 and face_detected:
            uart_communicator.send_open_signal_to_esp32()
            gen_frames.last_send_time = current_time

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        # Convert the frame to bytes
        processed_frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')

    camera.release()

@app.route('/')
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
    data = request.json
    if 'image' in data:
        # Extract the base64-encoded image data
        image_data = data['image'].split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
        # Save the image to a file with provided name
        image.save(f"./faces/{user_name}.png")
        model_services.build_target_for_user(user_name)

        return jsonify({"message": f"Image for {user_name} received and saved successfully!"})
    else:
        return jsonify({"message": "No image data received"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, ssl_context=('cert.pem', 'key.pem'))
