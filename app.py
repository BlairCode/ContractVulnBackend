"""
SCANA Flask Application

This service allows users to register, log in, and submit code for vulnerability scanning.
A machine learning model (BLSTM + attention) predicts potential vulnerabilities in the submitted code.
Results are stored in a SQLite database and can be queried for status and history.

Enhancements:
- Full, detailed comments
- Richer task information for frontend display
- Robust error handling and logging
"""

from flask import Flask, request, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from threading import Thread, Lock
from flask_cors import CORS
import time
import uuid
import datetime
import os
import torch
import traceback
from model.model import Fusion_Model_BLSTM_ATT
from model.w2v.model import get_embd

# ===========================
# Model Configuration
# ===========================
CHECKPOINT_PATH = os.path.join('model', 'w2v', 'checkpoints', 'blstm_epoch470.pt')
W2V_CP_PATH = os.path.join('model', 'w2v', 'checkpoints', 'sg_epoch1700.pt')

print("[INFO] Loading model...")
model = Fusion_Model_BLSTM_ATT(w2v_cp=W2V_CP_PATH, device='cpu', inference=True)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
model.eval()
print("[INFO] Model loaded successfully.")

# ===========================
# Flask App Initialization
# ===========================
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scana.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'e3f5a7b9c2d4e6f8091a2b3c4d5e6f70'
CORS(app, supports_credentials=True)
db = SQLAlchemy(app)

# ===========================
# Database Models
# ===========================
class User(db.Model):
    """Database model for registered users."""
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

class ScanTask(db.Model):
    """Database model for scan tasks."""
    __tablename__ = 'scan_task'
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    code = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='pending')
    progress = db.Column(db.Float, default=0.0)
    lines_scanned = db.Column(db.Integer, default=0)
    estimated_time_left = db.Column(db.Integer, default=-1)
    vul_distribution = db.Column(db.JSON, default={})
    vul_list = db.Column(db.JSON, default=[])
    start_time = db.Column(db.Float, nullable=True)
    end_time = db.Column(db.Float, nullable=True)
    error_msg = db.Column(db.String(255), nullable=True)

    user = db.relationship('User')

# Create database tables
with app.app_context():
    db.create_all()

# ===========================
# In-memory Task Management
# ===========================
scan_tasks = {}  # Cache for task info
scan_tasks_lock = Lock()  # Lock to synchronize access

# ===========================
# Core Scanning Function
# ===========================
def real_scan(task_id):
    """
    Background task to scan code for vulnerabilities.

    Workflow:
    1. Update task status to 'running'
    2. Generate code embeddings
    3. Run ML model to predict vulnerabilities
    4. Update in-memory cache and database
    5. Handle errors gracefully
    """
    print(f"[real_scan] Task {task_id} started.")
    with app.app_context():
        db_task = ScanTask.query.get(task_id)

        # Mark task as running in memory
        with scan_tasks_lock:
            scan_tasks[task_id]['status'] = 'running'
            scan_tasks[task_id]['start_time'] = time.time()

        # Update database task status
        if db_task:
            db_task.status = 'running'
            db_task.start_time = time.time()
            db.session.commit()

        code = scan_tasks[task_id].get('code', '')
        if not code.strip():
            err_msg = "Empty code"
            print(f"[real_scan] Task {task_id} failed: {err_msg}")
            with scan_tasks_lock:
                scan_tasks[task_id].update({'status': 'failed', 'error_msg': err_msg})
            if db_task:
                db_task.status = 'failed'
                db_task.error_msg = err_msg
                db.session.commit()
            return

        try:
            # Generate code embeddings
            embd_tensor = get_embd(code, 1700)
            if embd_tensor is None or embd_tensor.shape[0] == 0:
                raise ValueError("Empty or invalid code embedding")

            embd_tensor = embd_tensor.unsqueeze(0).float()
            print(f"[real_scan] Embedding shape: {embd_tensor.shape}, dtype: {embd_tensor.dtype}")

            # Model prediction
            with torch.no_grad():
                logits = model(embd_tensor)
                probs = torch.softmax(logits, dim=1)
                vul_prob = probs[0,1].item()
                print(f"[real_scan] Vulnerability probability: {vul_prob:.4f}")

            # Generate vulnerability report
            THRESHOLD = 0.2
            vul_list = []
            vul_distribution = {}
            if vul_prob >= THRESHOLD:
                vul = {
                    "id": str(uuid.uuid4()),
                    "line": 1,  # Placeholder for line-level info
                    "type": "predicted_vulnerability",
                    "severity": "high" if vul_prob > 0.8 else "medium",
                    "description": f"Predicted vulnerability with probability {vul_prob:.2f}"
                }
                vul_list.append(vul)
                vul_distribution["predicted_vulnerability"] = 1

            # Calculate scan duration
            end_time = time.time()
            duration = round(end_time - scan_tasks[task_id]['start_time'], 2)

            # Update in-memory cache
            with scan_tasks_lock:
                scan_tasks[task_id].update({
                    'status': 'done',
                    'progress': 1.0,
                    'lines_scanned': code.count('\n') + 1,
                    'estimated_time_left': 0,
                    'vul_list': vul_list,
                    'vul_distribution': vul_distribution,
                    'vul_prob': vul_prob,
                    'embedding_shape': list(embd_tensor.shape),
                    'duration': duration,
                    'error_msg': None
                })

            # Update database
            if db_task:
                db_task.status = 'done'
                db_task.progress = 1.0
                db_task.lines_scanned = code.count('\n') + 1
                db_task.estimated_time_left = 0
                db_task.vul_list = vul_list
                db_task.vul_distribution = vul_distribution
                db_task.end_time = end_time
                db_task.error_msg = None
                db.session.commit()

            print(f"[real_scan] Task {task_id} completed successfully in {duration}s.")

        except Exception as e:
            err_msg = str(e)
            print(f"[real_scan] Task {task_id} failed: {err_msg}")
            traceback.print_exc()
            with scan_tasks_lock:
                scan_tasks[task_id].update({'status': 'failed', 'error_msg': err_msg})
            if db_task:
                db_task.status = 'failed'
                db_task.error_msg = err_msg
                db.session.commit()

# ===========================
# Helper Functions
# ===========================
def get_current_user():
    """Return currently logged-in user or None if not logged in."""
    uid = session.get('user_id')
    return User.query.get(uid) if uid else None

def serialize_task(task: ScanTask):
    """
    Convert ScanTask object into a dictionary with rich info for frontend display:
    - Status, progress, scanned lines
    - Vulnerability list and distribution
    - Scan duration, embedding shape, predicted probability
    - Error messages
    """
    return {
        "task_id": task.id,
        "filename": task.filename,
        "status": task.status,
        "progress": round(task.progress, 3),
        "lines_scanned": task.lines_scanned,
        "estimated_time_left": task.estimated_time_left,
        "vul_distribution": task.vul_distribution,
        "vul_list": task.vul_list,
        "vul_prob": getattr(scan_tasks.get(task.id, {}), 'vul_prob', None),
        "embedding_shape": getattr(scan_tasks.get(task.id, {}), 'embedding_shape', None),
        "start_time": datetime.datetime.fromtimestamp(task.start_time).isoformat() if task.start_time else None,
        "end_time": datetime.datetime.fromtimestamp(task.end_time).isoformat() if task.end_time else None,
        "duration": f"{round((task.end_time - task.start_time), 2)}s" if task.start_time and task.end_time else None,
        "error_msg": task.error_msg
    }

# ===========================
# API Endpoints
# ===========================
@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.json
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Missing username or password"}), 400
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"error": "Username already exists"}), 400
    user = User(username=data['username'], password_hash=generate_password_hash(data['password']))
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Registration successful"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    """Login an existing user."""
    data = request.json
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Missing credentials"}), 400
    user = User.query.filter_by(username=data['username']).first()
    if not user or not check_password_hash(user.password_hash, data['password']):
        return jsonify({"error": "Invalid username or password"}), 401
    session['user_id'] = user.id
    return jsonify({"message": "Login successful", "username": user.username}), 200

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout the current user."""
    session.pop('user_id', None)
    return jsonify({"message": "Logged out successfully"}), 200

@app.route('/api/scan/start', methods=['POST'])
def start_scan():
    """Start a new scan task, supports file upload or direct code input."""
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        filename = file.filename
        code = file.read().decode('utf-8', errors='ignore')
    else:
        data = request.json
        if not data or 'filename' not in data or 'code' not in data:
            return jsonify({"error": "Missing filename or code"}), 400
        filename = data['filename']
        code = data['code']

    task_id = str(uuid.uuid4())

    task = ScanTask(
        id=task_id,
        user_id=user.id,
        filename=filename,
        code=code
    )
    db.session.add(task)
    db.session.commit()

    with scan_tasks_lock:
        scan_tasks[task_id] = {
            'status': 'pending',
            'progress': 0.0,
            'lines_scanned': 0,
            'estimated_time_left': -1,
            'vul_distribution': {},
            'vul_list': [],
            'filename': filename,
            'user_id': user.id,
            'code': code
        }

    Thread(target=real_scan, args=(task_id,), daemon=True).start()
    return jsonify({"message": "Scan started", "task_id": task_id}), 202

@app.route('/api/scan/status/<task_id>', methods=['GET'])
def scan_status(task_id):
    """Query the status of a scan task."""
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    task = ScanTask.query.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    if task.user_id != user.id:
        return jsonify({"error": "Access denied"}), 403
    return jsonify(serialize_task(task)), 200

@app.route('/api/scan/history', methods=['GET'])
def scan_history():
    """Query all scan history for the current user."""
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    tasks = ScanTask.query.filter_by(user_id=user.id).order_by(ScanTask.start_time.desc()).all()
    return jsonify([serialize_task(t) for t in tasks]), 200

# ===========================
# Static File Routes
# ===========================
@app.route('/')
def serve_index():
    """Serve the frontend index page."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve other frontend static files."""
    return send_from_directory(app.static_folder, path)

# ===========================
# App Entry Point
# ===========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
