from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
from options import TrainOptions
import RPTC
import process
import time

model_path = "RPTC.pth"
opt = TrainOptions().parse(print_option=False)
model = RPTC.Net()
model.apply(RPTC.initWeight)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['netC'].items()})
model.eval()


def predict(img_path, opt):
    img = Image.open(img_path).convert('RGB')
    input_tensor = process.processing_RPTC(img, opt).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        result = torch.sigmoid(output).item()

    print(img_path)
    print(f'AI-generated probability: {round(result*100, 3)}%')
    prob = round(result*100, 3)
    return prob


app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def serve_frontend():
    return send_from_directory(app.static_folder, 'ai_detector_spa.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # predict(filepath, opt)
        return jsonify({'message': 'File uploaded successfully', 'filepath': filepath}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400


@app.route('/detect', methods=['POST', 'GET'])
def detect_image():
    data = request.get_json()
    image_path = data.get('image_path')

    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Invalid or missing image path'}), 400

    try:
        # time.sleep(1.5)
        prob = predict(image_path, opt)
        return jsonify({'probability': prob})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
