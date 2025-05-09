from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
from PIL import Image
import torch
from options import TrainOptions
import RPTC
import process
import time
import io

model_path = "RPTC.pth"
opt = TrainOptions().parse(print_option=False)
model = RPTC.Net()
model.apply(RPTC.initWeight)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['netC'].items()})
model.eval()


def predict(img, opt):
    input_tensor = process.processing_RPTC(img, opt).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        result = torch.sigmoid(output).item()

    print(f'AI-generated probability: {round(result*100, 3)}%')
    prob = round(result*100, 3)
    return prob


app = Flask(__name__, static_folder='static')
CORS(app)


@app.route('/', methods=['POST', 'GET'])
def serve_frontend():
    return send_from_directory(app.static_folder, 'ai_detector_spa.html')


@app.route('/detect', methods=['POST', 'GET'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image upload'}), 400

    image_file = request.files['image']

    try:
        # time.sleep(1.5)
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        prob = predict(image, opt)
        return jsonify({'probability': prob}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
