# Author: BLAIR
# Backend interface for smart contract vulnerability detection.

import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import socket
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging to file and console
LOG_FILE = os.getenv("LOG_FILE", "app.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

# Import custom model modules
from model.w2v.build_vocab import tokenize
from model.w2v.model import SkipGram, get_embd
from model.model import Fusion_Model_BLSTM_ATT

# Select device for PyTorch (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logging.info(f"Using device: {device}")

# Initialize Flask app with static folder and enable CORS
app = Flask(__name__, static_folder=os.path.abspath(os.path.dirname(__file__)))
CORS(app)

# Set project root and upload folder
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(PROJECT_ROOT, "uploads"))
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logging.info(f"Created upload directory: {UPLOAD_FOLDER}")

# Load SkipGram model
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", os.path.join(PROJECT_ROOT, "model", "w2v", "sg_epoch0.pt"))
if not os.path.exists(MODEL_CHECKPOINT):
    logging.error(f"Model file not found: {MODEL_CHECKPOINT}")
    raise FileNotFoundError(f"Missing model file: {MODEL_CHECKPOINT}")
try:
    state_dict = torch.load(MODEL_CHECKPOINT, map_location=device)
    vocab_size = state_dict['in_embed.weight'].shape[0]
    sg_model = SkipGram(vocab_size, 50).to(device)
    sg_model.load_state_dict(state_dict)
    sg_model.eval()
    logging.info(f"SkipGram model loaded: {MODEL_CHECKPOINT}, vocab size: {vocab_size}")
except Exception as e:
    logging.error(f"Failed to load SkipGram model: {str(e)}", exc_info=True)
    raise

# Load Fusion model
FUSION_CHECKPOINT = os.getenv("FUSION_CHECKPOINT", os.path.join(PROJECT_ROOT, "model", "fusion_model.pt"))
try:
    fusion_model = Fusion_Model_BLSTM_ATT(device=device, inference_only=True).to(device)
    if os.path.exists(FUSION_CHECKPOINT):
        fusion_model.load_state_dict(torch.load(FUSION_CHECKPOINT, map_location=device))
        fusion_model.eval()
        logging.info(f"Fusion model loaded: {FUSION_CHECKPOINT}")
    else:
        logging.warning(f"Fusion model checkpoint not found: {FUSION_CHECKPOINT}")
except Exception as e:
    logging.error(f"Failed to load Fusion model: {str(e)}", exc_info=True)
    raise

# Prediction function for vulnerability detection
def predict(fusion_model, input_data):
    fusion_model.eval()
    input_tensor = input_data.to(device).float()
    with torch.no_grad():
        logits = fusion_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities[0].cpu().numpy()

# Route to serve index.html
@app.route('/')
def index():
    try:
        return send_from_directory(PROJECT_ROOT, 'index.html')
    except Exception as e:
        logging.error(f"Failed to load index page: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to load index page"}), 500

# Route to handle file uploads
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        logging.info(f"File uploaded successfully: {filepath}")
        return jsonify({"message": "File uploaded successfully", "filepath": filepath})
    except PermissionError:
        logging.error(f"Permission denied to save file: {filepath}")
        return jsonify({"error": "Permission denied"}), 403
    except Exception as e:
        logging.error(f"File upload failed: {str(e)}", exc_info=True)
        return jsonify({"error": "File save failed", "details": str(e)}), 500

# Route to detect vulnerabilities in uploaded file
@app.route("/detect", methods=["POST"])
def detect_vulnerabilities():
    try:
        data = request.json
        file_path = data.get("filepath")
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 400
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read().replace('\n', ' ')
        tokens = tokenize(code)
        vecs = get_embd(tokens)
        if vecs is None:
            return jsonify({"error": "Failed to generate embedding vectors"}), 500
        input_data = vecs.clone().detach().unsqueeze(0).to(device)
        predicted_class, probabilities = predict(fusion_model, input_data)
        results = {"predicted_class": predicted_class, "probabilities": probabilities.tolist()}
        logging.info(f"Vulnerability detection completed: {file_path}, results: {results}")
        return jsonify({"filepath": file_path, "results": results})
    except Exception as e:
        logging.error(f"Vulnerability detection failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Detection failed", "details": str(e)}), 500

# Check if a port is available
def check_port(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            logging.error(f"Port {port} is already in use")
            return False

# Main entry point: Start Flask app with port fallback
if __name__ == "__main__":
    HOST = os.getenv("HOST", "0.0.0.0")  # Default to all interfaces
    PORT = int(os.getenv("PORT", 8888))  # Default port 8888
    max_attempts = 5  # Maximum number of port attempts
    for attempt in range(max_attempts):
        if check_port(HOST, PORT):
            app.run(host=HOST, port=PORT, debug=True)
            break
        else:
            PORT += 1  # Increment port if occupied
            logging.info(f"Trying port: {PORT}")
    else:
        raise SystemExit(f"Startup failed: No available port found after {max_attempts} attempts")
