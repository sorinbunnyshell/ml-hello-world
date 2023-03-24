from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
import threading
from PIL import Image
from io import BytesIO
import os
import base64
import time

from models.cifar10_model import CIFAR10Model
from models.cifar10_training import start_training
from utils.locks import training_lock

# Read the allowed origin from the environment variable
allowed_origin = os.environ.get("ALLOWED_ORIGIN", "http://localhost")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": allowed_origin}})


model = CIFAR10Model()

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    json_data = request.get_json()
    image_data = json_data['image']
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data)).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = preprocess(image)

    prediction = model.predict(image_tensor)
    elapsed_time = time.time() - start_time
    return jsonify({'prediction': prediction, 'time': elapsed_time})

@app.route('/start-training', methods=['POST'])
def start_training_route():
    if training_lock.locked():
        return jsonify({'status': 'failed', 'message': 'Training already in progress'})

    # Get the number of epochs from the request JSON
    json_data = request.get_json()
    epochs = json_data.get('epochs', 10)  # Default to 10 epochs if not specified

    # Start training in a separate thread
    training_thread = threading.Thread(target=start_training, args=(epochs,))
    training_thread.start()
    return jsonify({'status': 'success', 'message': 'Training started'})

@app.route('/check-training-status', methods=['GET'])
def check_training_status():
    status = 'finished' if not training_lock.locked() else 'in progress'
    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
