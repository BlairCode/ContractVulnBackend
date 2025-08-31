"""
SCANA Enhanced Flask Application

This service provides a robust and scalable platform for smart contract vulnerability scanning.
Users can register, log in, and submit Solidity code for analysis. The backend uses a 
pre-trained BLSTM model with an attention mechanism to predict potential vulnerabilities.

Key Architectural Enhancements:
- Professional, human-readable comments and docstrings.
- Comprehensive logging to timestamped files instead of stdout.
- Centralized application configuration for better environment management.
- Scalable file handling: uploaded code is saved to disk, not the database.
- Improved database schema with more detailed task information.
- Enhanced API responses for a richer frontend experience.
- Robust error handling and database transaction management.
"""

import os
import uuid
import time
import logging
import traceback
from logging.handlers import RotatingFileHandler
from threading import Thread, Lock
from datetime import datetime

import torch
from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from flask import send_from_directory

# Assuming the model files are structured as in the original prompt.
from model.model import Fusion_Model_BLSTM_ATT
from model.w2v.model import get_embd

# ============================================================================
# >> 1. CONFIGURATION SETUP
# ============================================================================
# Centralized configuration makes the app easier to manage and deploy.
class Config:
    """Holds configuration variables for the Flask app."""
    # Secret key for session management. CHANGE THIS in a production environment.
    SECRET_KEY = os.environ.get('SECRET_KEY', 'e3f5a7b9c2d4e6f8091a2b3c4d5e6f70-you-should-change-this')
    
    # Database configuration.
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///scana.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Directory for storing uploaded source code files.
    UPLOADS_FOLDER = 'uploads'
    LOGS_FOLDER = 'logs'
    
    # Model checkpoints.
    CHECKPOINT_PATH = os.path.join('model', 'w2v', 'checkpoints', 'blstm_epoch470.pt')
    W2V_CP_PATH = os.path.join('model', 'w2v', 'checkpoints', 'sg_epoch1700.pt')

    # Vulnerability prediction threshold.
    VULNERABILITY_THRESHOLD = 0.2

# ============================================================================
# >> 2. LOGGING SETUP
# ============================================================================
# This function sets up a robust logger to replace all print() statements.
def setup_logging(app):
    """Configures file-based logging for the application."""
    if not os.path.exists(app.config['LOGS_FOLDER']):
        os.makedirs(app.config['LOGS_FOLDER'])

    # Create a unique log file for each run of the application.
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(app.config['LOGS_FOLDER'], f'scana_{timestamp}.log')

    # Configure the format for log messages.
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(threadName)s: %(message)s [in %(pathname)s:%(lineno)d]'
    )
    
    # Set up a file handler with rotation to prevent log files from getting too large.
    file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 5, backupCount=5) # 5 MB per file
    file_handler.setFormatter(formatter)
    
    # Set the logging level. In production, you might want to use logging.INFO.
    app.logger.setLevel(logging.DEBUG)
    app.logger.addHandler(file_handler)
    
    # Also log to console for development convenience.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    app.logger.addHandler(stream_handler)
    
    app.logger.info("SCANA application starting up...")
    app.logger.info(f"Logging configured. Log file at: {log_file}")

# ============================================================================
# >> 3. FLASK APP & EXTENSIONS INITIALIZATION
# ============================================================================
app = Flask(__name__)
app.config.from_object(Config)

# Initialize logging before doing anything else.
setup_logging(app)

# Ensure required directories exist.
os.makedirs(app.config['UPLOADS_FOLDER'], exist_ok=True)

# Initialize extensions.
db = SQLAlchemy(app)
CORS(app, supports_credentials=True)

# Global lock for thread-safe operations on the task cache.
# While we primarily rely on the database, a cache can be useful for very quick status checks.
scan_tasks_lock = Lock()


# ============================================================================
# >> 4. DATABASE MODELS
# ============================================================================
# The database schema is defined here. It's more detailed than the original.
class User(db.Model):
    """Represents a registered user in the database."""
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.username}>'

class ScanTask(db.Model):
    """Represents a single vulnerability scan task."""
    __tablename__ = 'scan_task'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    
    # --- Task Info ---
    filename = db.Column(db.String(255), nullable=False)
    # Store the path to the code file, not the code itself. This is much more scalable.
    code_path = db.Column(db.String(512), nullable=False)
    status = db.Column(db.String(20), default='pending', index=True)
    error_msg = db.Column(db.Text, nullable=True)

    # --- Progress & Timestamps ---
    progress = db.Column(db.Float, default=0.0)
    lines_scanned = db.Column(db.Integer, default=0)
    start_time = db.Column(db.DateTime, nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)

    # --- Results ---
    vul_distribution = db.Column(db.JSON, default=dict)
    vul_list = db.Column(db.JSON, default=list)
    # Storing the raw probability is useful for analytics and frontend display.
    vul_prob = db.Column(db.Float, nullable=True)
    # Storing the shape of the embedding can be useful for debugging the model.
    embedding_shape = db.Column(db.JSON, nullable=True)

    user = db.relationship('User', backref=db.backref('scan_tasks', lazy=True))

    def __repr__(self):
        return f'<ScanTask {self.id} ({self.filename})>'

# Create database tables within the application context.
with app.app_context():
    db.create_all()

# ============================================================================
# >> 5. MACHINE LEARNING MODEL LOADING
# ============================================================================
# Encapsulate model loading to keep the global namespace clean.
def load_ml_model(app_logger):
    """Loads the pre-trained PyTorch model into memory."""
    try:
        app_logger.info("Initializing ML model...")
        model = Fusion_Model_BLSTM_ATT(w2v_cp=app.config['W2V_CP_PATH'], device='cpu', inference=True)
        app_logger.info(f"Loading model state from checkpoint: {app.config['CHECKPOINT_PATH']}")
        model.load_state_dict(torch.load(app.config['CHECKPOINT_PATH'], map_location='cpu'))
        model.eval() # Set the model to evaluation mode.
        app_logger.info("ML model loaded successfully.")
        return model
    except FileNotFoundError as e:
        app_logger.error(f"Model checkpoint file not found: {e}. The application cannot perform scans.")
        return None
    except Exception as e:
        app_logger.error(f"An unexpected error occurred while loading the ML model: {e}")
        app_logger.error(traceback.format_exc())
        return None

# Load the model once at startup.
ml_model = load_ml_model(app.logger)


# ============================================================================
# >> 6. CORE SCANNING LOGIC
# ============================================================================
# This is the heart of the application, where the actual scan happens.
# It runs in a background thread to avoid blocking web requests.
def perform_scan_for_task(task_id):
    """
    Background worker function to execute a single scan task.
    This function contains the full lifecycle of a scan:
    1. Fetch task from DB.
    2. Read source code from file.
    3. Generate code embeddings.
    4. Run the ML model for prediction.
    5. Process results and update the database.
    6. Handle any errors that occur during the process.
    """
    app.logger.info(f"Starting background scan for task_id: {task_id}")
    # A new app context is needed to access the database from a different thread.
    with app.app_context():
        # Step 1: Fetch the task from the database.
        task = ScanTask.query.get(task_id)
        if not task:
            app.logger.error(f"Task {task_id} not found in database for scanning. Aborting.")
            return

        try:
            # Update status to 'running' in the database.
            task.status = 'running'
            task.start_time = datetime.utcnow()
            db.session.commit()
            
            # Step 2: Read the source code from the file.
            app.logger.info(f"Reading code from file: {task.code_path}")
            with open(task.code_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()

            if not code.strip():
                raise ValueError("Submitted code file is empty.")

            # Step 3: Generate code embeddings using the w2v model.
            app.logger.info(f"Generating embeddings for task {task_id}...")
            # The '1700' seems to be a hardcoded model version or epoch.
            embd_tensor = get_embd(code, 1700) 
            
            if embd_tensor is None or embd_tensor.shape[0] == 0:
                raise ValueError("Failed to generate a valid code embedding. The code may be invalid or too short.")
            
            # The model expects a batch dimension, so we add one.
            embd_tensor = embd_tensor.unsqueeze(0).float()
            task.embedding_shape = list(embd_tensor.shape)
            app.logger.info(f"Embedding generated for task {task_id}. Shape: {task.embedding_shape}")

            # Step 4: Run the ML model for prediction.
            if ml_model is None:
                raise RuntimeError("ML model is not available. Cannot perform scan.")

            with torch.no_grad(): # Disable gradient calculation for inference.
                logits = ml_model(embd_tensor)
                probs = torch.softmax(logits, dim=1)
                # We are interested in the probability of the 'vulnerable' class (index 1).
                vul_prob = probs[0, 1].item()
            
            app.logger.info(f"Task {task_id} prediction complete. Vulnerability probability: {vul_prob:.4f}")
            task.vul_prob = vul_prob
            
            # Step 5: Process results.
            vul_list = []
            vul_distribution = {"high": 0, "medium": 0, "low": 0} # A more descriptive distribution.
            
            if vul_prob >= app.config['VULNERABILITY_THRESHOLD']:
                severity = "high" if vul_prob > 0.8 else ("medium" if vul_prob > 0.5 else "low")
                vul = {
                    "id": str(uuid.uuid4()),
                    # NOTE: This model provides contract-level predictions, not line-specific ones.
                    # 'line: 1' is a placeholder to conform to a potential frontend expectation.
                    "line": 1,
                    "type": "predicted_reentrancy", # Example type, could be more generic.
                    "severity": severity,
                    "description": f"A potential vulnerability was detected with a confidence of {vul_prob:.2%}.",
                    "confidence": f"{vul_prob:.4f}"
                }
                vul_list.append(vul)
                vul_distribution[severity] += 1

            # Finalize task details and update the database.
            task.status = 'done'
            task.progress = 1.0
            task.lines_scanned = code.count('\n') + 1
            task.vul_list = vul_list
            task.vul_distribution = vul_distribution
            task.end_time = datetime.utcnow()
            # db.session.commit()
            
            duration = (task.end_time - task.start_time).total_seconds()
            app.logger.info(f"Scan for task {task_id} completed successfully in {duration:.2f}s.")

        except Exception as e:
            # If anything goes wrong, log the error and mark the task as 'failed'.
            app.logger.error(f"Scan for task {task_id} failed: {e}")
            app.logger.error(traceback.format_exc())
            task.status = 'failed'
            task.error_msg = str(e)
            task.end_time = datetime.utcnow()
            # db.session.commit()
        
        finally:
            # This commit saves the final state, whether 'done' or 'failed'.
            # Using a try/except for the commit itself adds another layer of robustness.
            try:
                db.session.commit()
            except Exception as db_err:
                app.logger.error(f"Database commit failed for task {task_id}: {db_err}")
                db.session.rollback()


# ============================================================================
# >> 7. HELPER FUNCTIONS
# ============================================================================
def get_current_user():
    """Retrieves the currently logged-in user from the session."""
    user_id = session.get('user_id')
    if not user_id:
        return None
    return User.query.get(user_id)

def serialize_task(task: ScanTask):
    """
    Converts a ScanTask database object into a JSON-serializable dictionary.
    This rich object provides all necessary information for the frontend.
    """
    duration = None
    if task.start_time and task.end_time:
        duration = f"{round((task.end_time - task.start_time).total_seconds(), 2)}s"
    elif task.status == 'running' and task.start_time:
        # Provide a running duration for tasks in progress.
        duration = f"{round((datetime.utcnow() - task.start_time).total_seconds(), 2)}s (running)"
        
    return {
        "task_id": task.id,
        "filename": task.filename,
        "status": task.status,
        "progress": round(task.progress, 3),
        "lines_scanned": task.lines_scanned,
        "vul_distribution": task.vul_distribution,
        "vul_list": task.vul_list,
        "vul_prob": task.vul_prob,
        "embedding_shape": task.embedding_shape,
        "start_time": task.start_time.isoformat() if task.start_time else None,
        "end_time": task.end_time.isoformat() if task.end_time else None,
        "duration": duration,
        "error_msg": task.error_msg
    }

# ============================================================================
# >> 8. API ENDPOINTS
# ============================================================================
# These are the routes that the frontend will interact with.

# @app.route('/api/register', methods=['POST'])
# def register():
#     """Registers a new user."""
#     data = request.get_json()
#     if not data or not data.get('username') or not data.get('password'):
#         return jsonify({"error": "Username and password are required."}), 400
    
#     username = data['username']
#     if User.query.filter_by(username=username).first():
#         return jsonify({"error": f"Username '{username}' already exists."}), 409 # 409 Conflict

#     new_user = User(
#         username=username,
#         password_hash=generate_password_hash(data['password'])
#     )
#     db.session.add(new_user)
#     db.session.commit()
#     app.logger.info(f"New user registered: {username}")
#     return jsonify({"message": "Registration successful."}), 201

# @app.route('/api/login', methods=['POST'])
# def login():
#     """Logs in an existing user and creates a session."""
#     data = request.get_json()
#     if not data or not data.get('username') or not data.get('password'):
#         return jsonify({"error": "Username and password are required."}), 400

#     user = User.query.filter_by(username=data['username']).first()
#     if user and check_password_hash(user.password_hash, data['password']):
#         session['user_id'] = user.id
#         app.logger.info(f"User '{user.username}' logged in successfully.")
#         return jsonify({"message": "Login successful.", "username": user.username}), 200
    
#     app.logger.warning(f"Failed login attempt for username: {data.get('username')}")
#     return jsonify({"error": "Invalid username or password."}), 401

# @app.route('/api/logout', methods=['POST'])
# def logout():
#     """Logs out the current user by clearing the session."""
#     user_id = session.pop('user_id', None)
#     if user_id:
#         app.logger.info(f"User with ID {user_id} logged out.")
#     return jsonify({"message": "Logged out successfully."}), 200

@app.route('/api/scan/start', methods=['POST'])
def start_scan():
    """
    Starts a new scan task. Accepts either a file upload ('file') or
    raw JSON payload with 'filename' and 'code'.
    """
    user = get_current_user()
    if not user:
        return jsonify({"error": "Authentication required."}), 401

    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400
        filename = file.filename
        code = file.read().decode('utf-8', errors='ignore')
    else:
        data = request.get_json()
        if not data or not data.get('filename') or not data.get('code'):
            return jsonify({"error": "Request must include 'filename' and 'code'."}), 400
        filename = data['filename']
        code = data['code']
    
    # Save the code to a file to handle large inputs and keep the DB light.
    task_id = str(uuid.uuid4())
    unique_filename = f"{task_id}_{filename}"
    code_path = os.path.join(app.config['UPLOADS_FOLDER'], unique_filename)
    try:
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)
    except IOError as e:
        app.logger.error(f"Failed to write uploaded file to disk: {e}")
        return jsonify({"error": "Failed to save submitted code."}), 500

    # Create the task record in the database.
    new_task = ScanTask(
        id=task_id,
        user_id=user.id,
        filename=filename,
        code_path=code_path
    )
    db.session.add(new_task)
    db.session.commit()
    
    # Start the scan in a background thread.
    scan_thread = Thread(target=perform_scan_for_task, args=(task_id,), name=f"ScanThread-{task_id[:8]}")
    scan_thread.daemon = True
    scan_thread.start()
    
    app.logger.info(f"Scan task {task_id} created for user '{user.username}' and filename '{filename}'.")
    return jsonify({"message": "Scan task initiated successfully.", "task_id": task_id}), 202

@app.route('/api/scan/status/<string:task_id>', methods=['GET'])
def scan_status(task_id):
    """Retrieves the status and results of a specific scan task."""
    user = get_current_user()
    if not user:
        return jsonify({"error": "Authentication required."}), 401

    task = ScanTask.query.get(task_id)
    if not task:
        return jsonify({"error": "Scan task not found."}), 404
    
    # Ensure users can only access their own tasks.
    if task.user_id != user.id:
        return jsonify({"error": "Access denied. You do not own this task."}), 403

    return jsonify(serialize_task(task)), 200

@app.route('/api/scan/history', methods=['GET'])
def scan_history():
    """Retrieves the scan history for the currently logged-in user."""
    user = get_current_user()
    if not user:
        return jsonify({"error": "Authentication required."}), 401
    
    # Fetch tasks, newest first. Add pagination for production use (e.g., with .paginate()).
    tasks = ScanTask.query.filter_by(user_id=user.id).order_by(ScanTask.start_time.desc()).all()
    
    return jsonify([serialize_task(t) for t in tasks]), 200


# ============================================================================
# >> 9. FRONTEND SERVING
# ============================================================================
# These routes are necessary to serve the frontend application (e.g., a React or Vue app)
# The 'static_folder' is where Flask will look for the index.html and other assets.

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serves the frontend's static files."""
    static_folder = 'static'
    if path != "" and os.path.exists(os.path.join(static_folder, path)):
        return send_from_directory(static_folder, path)
    else:
        return send_from_directory(static_folder, 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)