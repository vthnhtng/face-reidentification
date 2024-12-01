from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
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

import os
from PIL import Image
import cv2

def excute_recognited_face(current_time, best_match_name, processed_frame):
    try:
        # Validate inputs
        if not isinstance(current_time, (int, float)):
            raise ValueError("Invalid type for 'current_time'. Expected int or float.")
        if not isinstance(best_match_name, str) or not best_match_name:
            raise ValueError("Invalid 'face_detected'. Must be a non-empty string.")
        if not isinstance(processed_frame, (list, tuple, np.ndarray)):
            raise ValueError("Invalid 'processed_frame'. Expected a NumPy array.")
        
        # Generate timestamp and filename
        timestamp = int(current_time * 1000)  # Convert seconds to milliseconds
        filename = f'{timestamp}_{best_match_name}.png'

        # Convert the processed frame to a PIL image
        image = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

        # Ensure the logs directory exists
        logs_directory = 'static/logs'
        if not os.path.exists(logs_directory):
            os.makedirs(logs_directory)

        log_files = [f for f in os.listdir(logs_directory) if os.path.isfile(os.path.join(logs_directory, f))]

        if len(log_files) > 20:
            log_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_directory, x)))
            for file_to_delete in log_files[:-19]:
                os.remove(os.path.join(logs_directory, file_to_delete))

        # Save the image
        file_path = os.path.join(logs_directory, filename)
        image.save(file_path)

        # Notify via UART (make sure uart_communicator is defined)
        if 'uart_communicator' in globals():
            uart_communicator.send_open_signal_to_esp32()
        else:
            print("Warning: 'uart_communicator' not defined. Skipping signal.")
        
        print(f"Image saved successfully: {file_path}")
    except Exception as e:
        print(f"Error in excute_recognited_face: {e}")


def gen_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        processed_frame, best_match_name = model_services.frame_process(frame)

        # Static variable to store the last time we sent a message
        if not hasattr(gen_frames, 'last_send_time'):
            gen_frames.last_send_time = 0

        # Get current time
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        # Check if 2 seconds have passed since last send
        if current_time - gen_frames.last_send_time >= 2.0 and best_match_name:
            excute_recognited_face(current_time, best_match_name, processed_frame)
            gen_frames.last_send_time = current_time

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        # Convert the frame to bytes
        processed_frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')

    camera.release()

app.secret_key = 'a3dcb8457e6341ef9a37f24c8b719e34'

users = {
    'admin': 'admin123',
}

def isLoggedIn():
    return 'user' in session

@app.route('/')
def index():
    session.pop('user', None)
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    images = os.listdir(image_folder)

    # Filter out non-image files if necessary (optional)
    images = [image for image in images if image.endswith(('png', 'jpg', 'jpeg', 'gif'))]

    return render_template('dashboard.html', images=images)

@app.route('/upload-image/<user_name>', methods=['POST'])
def upload_image(user_name):
    if not isLoggedIn():
        return jsonify({"message": f"You need login first"})

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
    app.run()
