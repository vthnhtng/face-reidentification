from flask import Flask, render_template, request, jsonify
# from model_services import ModelServices
# import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# model_services = ModelServices()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    data = request.json
    if 'image' in data:
        # Extract the base64-encoded image data
        image_data = data['image'].split(',')[1]  # Remove the "data:image/png;base64," prefix
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
        # Save the image to a file
        image.save("captured_image.png")
        
        return jsonify({"message": "Image received and saved successfully!"})
    else:
        return jsonify({"message": "No image data received"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
