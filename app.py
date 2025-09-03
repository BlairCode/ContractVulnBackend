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

# PDF生成相关包
from weasyprint import HTML, CSS
import pdfkit
import json
import shutil
import requests

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

class Job(db.Model):
    """
    一个完整的智能合约分析作业,包含LLM分析、ML推理、融合等多个阶段。
    作业状态机:PENDING → PREPROCESSING → LLM_ANALYZING → ML_INFER → FUSING → REPORT_READY / FAILED
    """
    __tablename__ = 'job'
    
    # 基本信息
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    upload_id = db.Column(db.String(36), nullable=False, index=True)  # 关联的上传ID
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)  # 用户ID
    
    # 作业配置
    slice_kind = db.Column(db.String(50), default='full')  # 切片类型：full, function, statement等
    llm_model = db.Column(db.String(100), default='deepseek')  # LLM模型：deepseek, gpt-4等
    notes = db.Column(db.Text, nullable=True)  # 用户备注
    
    # 状态与进度
    status = db.Column(db.String(20), default='PENDING', index=True)  # 当前状态
    current_stage = db.Column(db.String(20), default='PENDING')  # 当前阶段
    progress = db.Column(db.Float, default=0.0)  # 整体进度 0.0-1.0
    stage_progress = db.Column(db.Float, default=0.0)  # 当前阶段进度 0.0-1.0
    error_msg = db.Column(db.Text, nullable=True)  # 错误信息
    
    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    finished_at = db.Column(db.DateTime, nullable=True)
    
    # 产物路径（相对于 uploads/{upload_id}/）
    source_path = db.Column(db.String(512), nullable=True)  # 源文件路径
    preproc_path = db.Column(db.String(512), nullable=True)  # 预处理产物路径
    llm_result_path = db.Column(db.String(512), nullable=True)  # LLM分析结果路径
    ml_result_path = db.Column(db.String(512), nullable=True)  # ML推理结果路径
    fusion_result_path = db.Column(db.String(512), nullable=True)  # 融合结果路径
    html_report_path = db.Column(db.String(512), nullable=True)  # HTML报告路径
    pdf_report_path = db.Column(db.String(512), nullable=True)  # PDF报告路径
    
    # 结果摘要
    final_score = db.Column(db.Float, nullable=True)  # 最终风险评分 0.0-1.0
    severity = db.Column(db.String(20), nullable=True)  # 严重性：Critical/High/Medium/Low
    vulnerability_count = db.Column(db.Integer, default=0)  # 发现的漏洞数量
    
    # 关系
    user = db.relationship('User', backref=db.backref('jobs', lazy=True))
    
    def __repr__(self):
        return f'<Job {self.id} ({self.status})>'

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

# ================= 作业状态机与融合算法相关 =====================

# 作业状态枚举
JOB_STATES = {
    'PENDING': '等待开始',
    'PREPROCESSING': '预处理中',
    'LLM_ANALYZING': 'LLM分析中',
    'ML_INFER': 'ML推理中',
    'FUSING': '结果融合中',
    'REPORT_READY': '报告已生成',
    'FAILED': '失败'
}

# 融合算法配置
FUSION_CONFIG = {
    # 场景：跨境贸易权重
    'alpha': 0.35,  # ML概率权重
    'beta': 0.45,   # LLM整体评分权重
    'gamma': 0.20,  # 问题聚合权重
    'conflict_threshold': 0.28,  # 冲突检测阈值
    'severity_thresholds': {  # 严重性映射阈值
        'Critical': 0.8,
        'High': 0.6,
        'Medium': 0.4,
        'Low': 0.0
    }
}

def update_job_status(job_id, status, stage=None, progress=None, stage_progress=None, error_msg=None):
    """
    更新作业状态的通用函数
    
    Args:
        job_id: 作业ID
        status: 新状态
        stage: 当前阶段（可选）
        progress: 整体进度（可选）
        stage_progress: 阶段进度（可选）
        error_msg: 错误信息（可选）
    """
    try:
        job = Job.query.get(job_id)
        if not job:
            app.logger.error(f"作业 {job_id} 不存在")
            return False
            
        job.status = status
        if stage:
            job.current_stage = stage
        if progress is not None:
            job.progress = progress
        if stage_progress is not None:
            job.stage_progress = stage_progress
        if error_msg:
            job.error_msg = error_msg
            
        if status == 'FAILED':
            job.finished_at = datetime.utcnow()
        elif status == 'REPORT_READY':
            job.finished_at = datetime.utcnow()
            job.progress = 1.0
            
        db.session.commit()
        app.logger.info(f"作业 {job_id} 状态更新为: {status}")
        return True
    except Exception as e:
        app.logger.error(f"更新作业状态失败: {e}")
        db.session.rollback()
        return False

def fuse_llm_ml_results(llm_result, ml_result):
    """
    融合LLM和ML分析结果的核心算法
    
    Args:
        llm_result: LLM分析结果(JSON)
        ml_result: ML推理结果(包含vul_prob等)
    
    Returns:
        融合后的结果字典
    """
    try:
        # 提取ML概率
        ml_prob = ml_result.get('vul_prob', 0.0)
        
        # 解析LLM结果，计算整体评分
        if isinstance(llm_result, str):
            try:
                llm_data = json.loads(llm_result)
            except:
                llm_data = []
        else:
            llm_data = llm_result
            
        # 如果LLM结果不是数组，包装成数组
        if not isinstance(llm_data, list):
            llm_data = [llm_data] if llm_data else []
        
        # 计算LLM整体评分（基于严重性和置信度）
        llm_overall = 0.0
        issue_count = len(llm_data)
        
        if issue_count > 0:
            severity_scores = {'Critical': 1.0, 'High': 0.8, 'Medium': 0.5, 'Low': 0.2}
            total_score = 0.0
            
            for issue in llm_data:
                severity = issue.get('severity', 'Low')
                confidence = float(issue.get('confidence', 0.5))
                severity_score = severity_scores.get(severity, 0.2)
                total_score += severity_score * confidence
                
            llm_overall = min(total_score / issue_count, 1.0)
        
        # 问题聚合评分（基于问题数量）
        issue_aggregation = min(issue_count * 0.1, 1.0)
        
        # 固定使用跨境贸易权重（唯一场景）
        alpha = FUSION_CONFIG['alpha']
        beta = FUSION_CONFIG['beta']
        gamma = FUSION_CONFIG['gamma']
        conflict_threshold = FUSION_CONFIG['conflict_threshold']

        # 融合计算：final_score = α*ML_prob + β*LLM_overall + γ*issue_aggregation
        final_score = (
            alpha * ml_prob +
            beta * llm_overall +
            gamma * issue_aggregation
        )
        final_score = min(final_score, 1.0)
        
        # 严重性映射
        severity = 'Low'
        for sev, threshold in sorted(FUSION_CONFIG['severity_thresholds'].items(), 
                                   key=lambda x: x[1], reverse=True):
            if final_score >= threshold:
                severity = sev
                break
        
        # 冲突检测
        conflict_detected = abs(ml_prob - llm_overall) > conflict_threshold
        
        # 支持证据收集
        supporting_evidence = []
        for issue in llm_data:
            evidence = {
                'type': issue.get('vuln_type', 'Unknown'),
                'location': issue.get('location', 'N/A'),
                'evidence': issue.get('evidence', ''),
                'severity': issue.get('severity', 'Low'),
                'confidence': issue.get('confidence', 0.5)
            }
            supporting_evidence.append(evidence)
        
        # 关键发现（取前5个最严重的）
        top_findings = sorted(llm_data, 
                            key=lambda x: (
                                FUSION_CONFIG['severity_thresholds'].get(x.get('severity', 'Low'), 0),
                                float(x.get('confidence', 0))
                            ), reverse=True)[:5]
        
        # 构建融合结果
        fusion_result = {
            'final_score': round(final_score, 4),
            'severity': severity,
            'vulnerability_count': issue_count,
            'ml_probability': ml_prob,
            'llm_overall_score': llm_overall,
            'conflict_detected': conflict_detected,
            'conflict_reason': f"ML概率({ml_prob:.3f})与LLM评分({llm_overall:.3f})差异过大" if conflict_detected else None,
            'top_findings': top_findings,
            'supporting_evidence': supporting_evidence,
            'fusion_metadata': {
                'alpha': alpha,
                'beta': beta, 
                'gamma': gamma,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        return fusion_result
        
    except Exception as e:
        app.logger.error(f"融合算法执行失败: {e}")
        return {
            'final_score': 0.0,
            'severity': 'Low',
            'vulnerability_count': 0,
            'error': str(e)
        }

# ================= LLM 配置与调用相关 =====================
# deepseek LLM 配置，支持 API Key、Endpoint、超时、重试，参数从环境变量读取
LLM_API_KEY = os.environ.get('LLM_API_KEY', '')
LLM_API_ENDPOINT = os.environ.get('LLM_API_ENDPOINT', 'https://api.deepseek.com/v1/chat/completions')
LLM_TIMEOUT = int(os.environ.get('LLM_TIMEOUT', '60'))
LLM_MAX_RETRY = int(os.environ.get('LLM_MAX_RETRY', '3'))
LLM_MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '2048'))

# LLM Prompt 设计（精心设计的系统提示、指令、few-shot示例以及JSON Schema，聚焦跨境贸易合规场景）
LLM_SYSTEM_PROMPT = """
[系统提示]
你是资深的跨境贸易智能合约安全审计专家与合规顾问，熟悉跨境贸易业务逻辑、清关流程、结算与合规（KYC/AML、制裁名单筛查、数据跨境传输与数据最小化要求），精通 Solidity 语言与安全标准（OWASP/SWC/CWE）。
你的目标：
- 针对跨境贸易场景识别并解释智能合约中的潜在安全漏洞与合规风险；
- 提供专业、可执行的修复建议与合规建议；
- 输出结构化、可被后端解析的 JSON；
- 在 standard 字段中指向跨境贸易相关合规点（如 KYC、AML、制裁名单、数据跨境传输合规等）。

[指令提示]
- 严格遵循输出 JSON Schema，不要输出除 JSON 外的内容。
- 若无法确定，请给出保守判断，并降低置信度。
- 在 location 字段中尽可能给出函数名与行号范围。
- evidence 提供关键最小代码片段；suggestion 给出操作性措施（权限控制、输入校验、ReentrancyGuard、事件审计、分离结算与业务逻辑等）。

[输出 JSON Schema]
严格按照以下 JSON Schema 输出，如有多个发现请输出 JSON 数组：
[
  {
    "vuln_type": "字符串，漏洞类型（如 Reentrancy、Access Control、Integer Overflow 等）",
    "location": "字符串，函数名与/或行号范围（如 settle(): L40-L65）",
    "evidence": "字符串，最小可支持的代码片段或关键语句",
    "severity": "枚举：Critical|High|Medium|Low",
    "confidence": "0.0-1.0 的字符串或数值，代表置信度",
    "suggestion": "字符串，工程化修复建议（如使用ReentrancyGuard/严格权限控制/额度与限额校验/可追溯事件日志）",
    "standard": "字符串，适用标准或合规要求（SWC/CWE/OWASP/KYC/AML/制裁名单筛查/数据跨境合规点）",
    "cwe_id": "字符串，可选，如 CWE-841",
    "impact": "字符串，潜在影响描述（如资金被盗、拒付、制裁风险、数据违规传输等）"
  }
]

[few-shot 示例]
示例输入（节选）：
contract Settlement {
    mapping(address => uint256) balance;
    function withdraw(uint256 amount) external {
        require(balance[msg.sender] >= amount, "insufficient");
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok);
        balance[msg.sender] -= amount;
    }
}

示例输出：
[
  {
    "vuln_type": "Reentrancy",
    "location": "withdraw(): L4-L9",
    "evidence": "外部调用 call{value: amount} 在状态更新之前执行，且缺少重入保护",
    "severity": "Critical",
    "confidence": "0.95",
    "suggestion": "采用CEI模式或使用ReentrancyGuard；对大额提现增加额度/限额校验并记录审计事件",
    "standard": "SWC-107, KYC/AML(提现风控), OWASP Smart Contract Top 10",
    "cwe_id": "CWE-841",
    "impact": "攻击者可重入导致资金被盗，跨境结算存在洗钱与违规转移风险"
  }
]

[开始分析]
请基于跨境贸易合规场景分析以下Solidity代码：
"""

# LLM 调用函数，带重试和超时

def call_deepseek_llm(code, stage='precheck'):
    """
    调用 deepseek LLM API,对合约代码进行漏洞分析（多阶段）。
    返回 LLM 的 JSON 结果字符串，失败时返回空字符串。
    """
    # 多阶段系统提示（围绕跨境贸易领域）
    STAGE_TO_PROMPT = {
        'precheck': LLM_SYSTEM_PROMPT,
        'deep_audit': LLM_SYSTEM_PROMPT + "\n[阶段加成]请在更严格的审计强度下，结合重入/访问控制/授权绕过/资产冻结与释放、清算与结算路径等高危风险，输出更完整的问题集合。",
        'compliance': LLM_SYSTEM_PROMPT + "\n[阶段加成]聚焦跨境贸易合规：KYC/AML、受制裁名单、地域与数据跨境传输限制、资金来源合规、额度与限额合规、可追溯与审计日志等，输出合规模块专项发现。"
    }
    system_prompt = STAGE_TO_PROMPT.get(stage, LLM_SYSTEM_PROMPT)
    headers = {
        'Authorization': f'Bearer {LLM_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': 'deepseek-chat',  # deepseek 官方模型名，可根据实际情况调整
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': code}
        ],
        'max_tokens': LLM_MAX_TOKENS,
        'temperature': 0.2
    }
    for attempt in range(LLM_MAX_RETRY):
        try:
            resp = requests.post(
                LLM_API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=LLM_TIMEOUT
            )
            if resp.status_code == 200:
                data = resp.json()
                # 兼容 deepseek 返回格式，提取 content
                content = data['choices'][0]['message']['content']
                return content
            else:
                app.logger.warning(f"LLM API 调用失败，状态码: {resp.status_code}, 内容: {resp.text}")
        except Exception as e:
            app.logger.warning(f"LLM API 调用异常: {e}")
    return ''

# ================ 上传后自动调用 LLM，保存初步分析结果 ================

from werkzeug.utils import secure_filename

@app.after_request
def auto_llm_analysis(response):
    """
    钩子：在 /api/uploads 上传成功后，自动调用 deepseek LLM 进行初步分析。
    结果保存到 uploads/{upload_id}/llm_result.json。
    """
    try:
        # 仅处理 /api/uploads 且上传成功
        if request.path == '/api/uploads' and response.status_code == 200:
            resp_json = response.get_json()
            upload_id = resp_json.get('upload_id')
            if upload_id:
                upload_dir = os.path.join(app.config['UPLOADS_FOLDER'], upload_id)
                # 查找合约文件（假设只有一个）
                files = [f for f in os.listdir(upload_dir) if f != 'meta.json']
                if files:
                    file_path = os.path.join(upload_dir, files[0])
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    # TODO: 敏感信息清洗/脱敏（可插入钩子）
                    # code = clean_sensitive_info(code)
                    llm_result = call_deepseek_llm(code)
                    # 尝试解析为 JSON，若失败则原样保存
                    try:
                        llm_json = json.loads(llm_result)
                    except Exception:
                        llm_json = llm_result
                    with open(os.path.join(upload_dir, 'llm_result.json'), 'w', encoding='utf-8') as f:
                        json.dump(llm_json, f, ensure_ascii=False, indent=2)
                    app.logger.info(f"LLM 初步分析已保存: {upload_dir}/llm_result.json")
    except Exception as e:
        app.logger.warning(f"自动 LLM 分析失败: {e}")
    return response

# ================= 作业处理工作流程 =====================

def process_job_workflow(job_id):
    """
    作业处理的主工作流程，在后台线程中执行
    状态机:PENDING → PREPROCESSING → LLM_ANALYZING → ML_INFER → FUSING → REPORT_READY / FAILED
    """
    app.logger.info(f"开始处理作业: {job_id}")
    
    with app.app_context():
        try:
            job = Job.query.get(job_id)
            if not job:
                app.logger.error(f"作业 {job_id} 不存在")
                return
                
            job.started_at = datetime.utcnow()
            
            # 阶段1: 预处理
            update_job_status(job_id, 'PREPROCESSING', 'PREPROCESSING', 0.1, 0.0)
            preprocessing_success = perform_preprocessing(job)
            if not preprocessing_success:
                update_job_status(job_id, 'FAILED', error_msg='预处理失败')
                return
            
            # 阶段2: LLM分析（如果还没有结果）
            update_job_status(job_id, 'LLM_ANALYZING', 'LLM_ANALYZING', 0.3, 0.0)
            llm_success = perform_llm_analysis(job)
            if not llm_success:
                update_job_status(job_id, 'FAILED', error_msg='LLM分析失败')
                return
                
            # 阶段3: ML推理
            update_job_status(job_id, 'ML_INFER', 'ML_INFER', 0.6, 0.0)
            ml_success = perform_ml_inference(job)
            if not ml_success:
                update_job_status(job_id, 'FAILED', error_msg='ML推理失败')
                return
                
            # 阶段4: 结果融合
            update_job_status(job_id, 'FUSING', 'FUSING', 0.8, 0.0)
            fusion_success = perform_fusion(job)
            if not fusion_success:
                update_job_status(job_id, 'FAILED', error_msg='结果融合失败')
                return
                
            # 阶段5: 报告生成
            update_job_status(job_id, 'FUSING', 'FUSING', 0.9, 0.8)
            report_success = generate_reports(job)
            if not report_success:
                update_job_status(job_id, 'FAILED', error_msg='报告生成失败')
                return
                
            # 完成
            update_job_status(job_id, 'REPORT_READY', 'REPORT_READY', 1.0, 1.0)
            app.logger.info(f"作业 {job_id} 处理完成")
            
        except Exception as e:
            app.logger.error(f"作业 {job_id} 处理异常: {e}")
            app.logger.error(traceback.format_exc())
            update_job_status(job_id, 'FAILED', error_msg=str(e))

def perform_preprocessing(job):
    """执行预处理阶段"""
    try:
        upload_dir = os.path.join(app.config['UPLOADS_FOLDER'], job.upload_id)
        preproc_dir = os.path.join(upload_dir, 'preproc')
        os.makedirs(preproc_dir, exist_ok=True)
        
        # 找到源文件
        files = [f for f in os.listdir(upload_dir) if f.endswith('.sol') or f != 'meta.json']
        if not files:
            app.logger.error(f"作业 {job.id} 未找到源文件")
            return False
            
        source_file = files[0]
        job.source_path = source_file
        
        # 这里可以添加更多预处理逻辑：
        # - ANTLR解析
        # - AST生成
        # - DOT图生成
        # - 代码切片
        
        # 简单复制源文件到预处理目录（占位符）
        source_path = os.path.join(upload_dir, source_file)
        preproc_source_path = os.path.join(preproc_dir, source_file)
        shutil.copy2(source_path, preproc_source_path)
        
        job.preproc_path = f'preproc/{source_file}'
        db.session.commit()
        
        app.logger.info(f"作业 {job.id} 预处理完成")
        return True
        
    except Exception as e:
        app.logger.error(f"预处理失败: {e}")
        return False

def perform_llm_analysis(job):
    """执行LLM分析阶段"""
    try:
        upload_dir = os.path.join(app.config['UPLOADS_FOLDER'], job.upload_id)
        llm_result_path = os.path.join(upload_dir, 'llm_result.json')
        
        # 检查是否已有LLM结果（上传时自动生成的）
        if os.path.exists(llm_result_path):
            job.llm_result_path = 'llm_result.json'
            db.session.commit()
            app.logger.info(f"作业 {job.id} 使用已有LLM分析结果")
            return True
        
        # 如果没有，重新执行LLM分析
        source_path = os.path.join(upload_dir, job.source_path)
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
            
        # 多阶段策略：先预筛选，再针对跨境合规进行专项，再进行深度审计
        stages = ['precheck', 'compliance', 'deep_audit']
        merged = []
        for st in stages:
            st_result = call_deepseek_llm(code, stage=st)
            if not st_result:
                continue
            try:
                items = json.loads(st_result)
                if isinstance(items, list):
                    merged.extend(items)
                elif isinstance(items, dict):
                    merged.append(items)
                else:
                    # 不可解析则忽略该阶段
                    pass
            except Exception:
                # 返回非JSON时忽略该阶段（保持健壮性）
                continue

        # 若多阶段均失败，回退一次单阶段
        if not merged:
            fallback = call_deepseek_llm(code, stage='precheck')
            try:
                merged = json.loads(fallback) if fallback else []
                if isinstance(merged, dict):
                    merged = [merged]
            except Exception:
                merged = []
        if not llm_result:
            return False
            
        # 保存LLM结果
        llm_json = merged
            
        with open(llm_result_path, 'w', encoding='utf-8') as f:
            json.dump(llm_json, f, ensure_ascii=False, indent=2)
            
        job.llm_result_path = 'llm_result.json'
        db.session.commit()
        
        app.logger.info(f"作业 {job.id} LLM分析完成")
        return True
        
    except Exception as e:
        app.logger.error(f"LLM分析失败: {e}")
        return False

def perform_ml_inference(job):
    """执行ML推理阶段"""
    try:
        upload_dir = os.path.join(app.config['UPLOADS_FOLDER'], job.upload_id)
        source_path = os.path.join(upload_dir, job.source_path)
        
        # 读取代码
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
            
        if not code.strip():
            raise ValueError("源代码文件为空")
            
        # 生成代码嵌入
        embd_tensor = get_embd(code, 1700)
        if embd_tensor is None or embd_tensor.shape[0] == 0:
            raise ValueError("无法生成有效的代码嵌入")
            
        embd_tensor = embd_tensor.unsqueeze(0).float()
        
        # ML模型推理
        if ml_model is None:
            raise RuntimeError("ML模型未加载")
            
        with torch.no_grad():
            logits = ml_model(embd_tensor)
            probs = torch.softmax(logits, dim=1)
            vul_prob = probs[0, 1].item()
            
        # 构建ML结果
        ml_result = {
            'vul_prob': vul_prob,
            'embedding_shape': list(embd_tensor.shape),
            'model_info': {
                'checkpoint_path': app.config['CHECKPOINT_PATH'],
                'w2v_checkpoint': app.config['W2V_CP_PATH']
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # 保存ML结果
        ml_result_path = os.path.join(upload_dir, 'ml_result.json')
        with open(ml_result_path, 'w', encoding='utf-8') as f:
            json.dump(ml_result, f, ensure_ascii=False, indent=2)
            
        job.ml_result_path = 'ml_result.json'
        db.session.commit()
        
        app.logger.info(f"作业 {job.id} ML推理完成，漏洞概率: {vul_prob:.4f}")
        return True
        
    except Exception as e:
        app.logger.error(f"ML推理失败: {e}")
        return False

def perform_fusion(job):
    """执行结果融合阶段"""
    try:
        upload_dir = os.path.join(app.config['UPLOADS_FOLDER'], job.upload_id)
        
        # 加载LLM结果
        llm_result_path = os.path.join(upload_dir, job.llm_result_path)
        with open(llm_result_path, 'r', encoding='utf-8') as f:
            llm_result = json.load(f)
            
        # 加载ML结果
        ml_result_path = os.path.join(upload_dir, job.ml_result_path)
        with open(ml_result_path, 'r', encoding='utf-8') as f:
            ml_result = json.load(f)
            
        # 执行融合算法
        fusion_result = fuse_llm_ml_results(llm_result, ml_result)
        
        # 保存融合结果
        fusion_result_path = os.path.join(upload_dir, 'fusion_result.json')
        with open(fusion_result_path, 'w', encoding='utf-8') as f:
            json.dump(fusion_result, f, ensure_ascii=False, indent=2)
            
        # 更新作业记录
        job.fusion_result_path = 'fusion_result.json'
        job.final_score = fusion_result.get('final_score', 0.0)
        job.severity = fusion_result.get('severity', 'Low')
        job.vulnerability_count = fusion_result.get('vulnerability_count', 0)
        db.session.commit()
        
        app.logger.info(f"作业 {job.id} 结果融合完成，最终评分: {job.final_score}")
        return True
        
    except Exception as e:
        app.logger.error(f"结果融合失败: {e}")
        return False

def generate_reports(job):
    """生成HTML和PDF报告"""
    try:
        upload_dir = os.path.join(app.config['UPLOADS_FOLDER'], job.upload_id)
        
        # 加载所有结果数据
        with open(os.path.join(upload_dir, job.fusion_result_path), 'r', encoding='utf-8') as f:
            fusion_result = json.load(f)
        
        with open(os.path.join(upload_dir, 'meta.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        # 生成HTML报告
        html_content = generate_html_report(job, fusion_result, meta)
        html_report_path = os.path.join(upload_dir, 'report.html')
        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        job.html_report_path = 'report.html'
        
        # 生成PDF报告
        pdf_report_path = os.path.join(upload_dir, 'report.pdf')
        pdf_success = generate_pdf_report(html_content, pdf_report_path)
        if pdf_success:
            job.pdf_report_path = 'report.pdf'
        else:
            app.logger.warning(f"作业 {job.id} PDF报告生成失败,但HTML报告可用")
        
        db.session.commit()
        
        app.logger.info(f"作业 {job.id} 报告生成完成")
        return True
        
    except Exception as e:
        app.logger.error(f"报告生成失败: {e}")
        return False

@app.route('/api/reports/sample', methods=['POST'])
def generate_sample_report():
    """
    生成一份示例报告（用于汇报/演示），领域限定为跨境贸易。
    - 自动创建临时 upload_id 目录
    - 生成 demo 的 fusion_result.json 与 meta.json
    - 渲染 HTML 与 PDF
    - 返回生成的相对路径
    """
    try:
        upload_id = str(uuid.uuid4())
        upload_dir = os.path.join(app.config['UPLOADS_FOLDER'], upload_id)
        os.makedirs(upload_dir, exist_ok=True)

        # 示例 meta（跨境贸易）
        meta = {
            'upload_id': upload_id,
            'filename': 'CrossBorderSettlement.sol',
            'business_domain': 'cross_border',
            'chain': 'ethereum',
            'compiler_version': '0.8.19',
            'notes': '示例报告：跨境贸易清结算合约',
            'upload_time': datetime.utcnow().isoformat()
        }
        with open(os.path.join(upload_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 示例融合结果（体现 LLM + ML 融合）
        fusion_result = {
            'final_score': 0.73,
            'severity': 'High',
            'vulnerability_count': 3,
            'ml_probability': 0.68,
            'llm_overall_score': 0.76,
            'conflict_detected': False,
            'conflict_reason': None,
            'top_findings': [
                {
                    'vuln_type': 'Reentrancy',
                    'location': 'withdraw(): L42-L70',
                    'evidence': 'call{value: amount} 在状态更新之前执行',
                    'severity': 'Critical',
                    'confidence': 0.92,
                    'suggestion': '采用CEI/加 ReentrancyGuard；提现额度风控与审计事件',
                    'standard': 'SWC-107, KYC/AML(提现风控)'
                },
                {
                    'vuln_type': 'Access Control',
                    'location': 'settle(): L10-L38',
                    'evidence': '管理操作缺少 onlyOwner/角色控制',
                    'severity': 'High',
                    'confidence': 0.81,
                    'suggestion': '基于角色的严格权限；多签审批',
                    'standard': 'OWASP, 合规审批链'
                },
                {
                    'vuln_type': 'Compliance-KYC/AML',
                    'location': 'transfer(): L85-L110',
                    'evidence': '未对收/付款方进行KYC与制裁名单校验',
                    'severity': 'Medium',
                    'confidence': 0.78,
                    'suggestion': '集成KYC/AML与制裁名单服务；限制高风险地域',
                    'standard': 'KYC/AML, Sanction Screening'
                }
            ],
            'supporting_evidence': [],
            'fusion_metadata': {
                'alpha': 0.35,
                'beta': 0.45,
                'gamma': 0.2,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        with open(os.path.join(upload_dir, 'fusion_result.json'), 'w', encoding='utf-8') as f:
            json.dump(fusion_result, f, ensure_ascii=False, indent=2)

        # 构造虚拟 Job 对象用于渲染（最小字段）
        class _J: pass
        job = _J()
        job.id = str(uuid.uuid4())
        job.upload_id = upload_id
        job.fusion_result_path = 'fusion_result.json'

        # 生成报告
        html_content = generate_html_report(job, fusion_result, meta)
        html_report_path = os.path.join(upload_dir, 'report.html')
        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        pdf_report_path = os.path.join(upload_dir, 'report.pdf')
        _ = generate_pdf_report(html_content, pdf_report_path)

        return jsonify({
            'message': '示例报告已生成',
            'upload_id': upload_id,
            'html_report_path': f'{upload_id}/report.html',
            'pdf_report_path': f'{upload_id}/report.pdf'
        }), 201
    except Exception as e:
        app.logger.error(f"示例报告生成失败: {e}")
        return jsonify({'error': f'示例报告生成失败: {str(e)}'}), 500

def generate_html_report(job, fusion_result, meta):
    """
    生成HTML报告内容（从静态模板渲染）
    - 模板文件位置：static/report_template.html
    - 通过占位符进行替换，保证模板与逻辑分离，便于前端单独维护
    占位符示例：
      {{FILENAME}}、{{JOB_ID}}、{{GENERATED_AT}}、{{FINAL_SCORE}}、{{SEVERITY}}、{{VULN_COUNT}}、
      {{ML_PROB}}、{{LLM_OVERALL}}、{{CONFLICT_DETECTED}}、{{CONFLICT_REASON}}、
      {{BUSINESS_DOMAIN}}、{{CHAIN}}、{{COMPILER_VERSION}}、{{NOTES}}、{{UPLOAD_TIME}}、
      {{TOP_FINDINGS_HTML}}
    """
    try:
        template_path = os.path.join('static', 'report_template.html')
        if not os.path.exists(template_path):
            raise FileNotFoundError('报告模板文件不存在: static/report_template.html')

        # 生成关键字段
        generated_at = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        final_score = f"{fusion_result.get('final_score', 0):.2f}"
        severity = fusion_result.get('severity', 'Low')
        vuln_count = str(fusion_result.get('vulnerability_count', 0))
        ml_prob = f"{fusion_result.get('ml_probability', 0):.4f}"
        llm_overall = f"{fusion_result.get('llm_overall_score', 0):.4f}"
        conflict_detected = '是' if fusion_result.get('conflict_detected') else '否'
        conflict_reason = fusion_result.get('conflict_reason', '') or ''

        top_findings_html = generate_findings_html(fusion_result.get('top_findings', []))

        # 读取模板
        with open(template_path, 'r', encoding='utf-8') as tf:
            html = tf.read()

        # 安全占位符替换（简单字符串替换，避免引入模板引擎依赖）
        replacements = {
            '{{FILENAME}}': meta.get('filename', 'Unknown'),
            '{{JOB_ID}}': job.id,
            '{{GENERATED_AT}}': generated_at,
            '{{FINAL_SCORE}}': final_score,
            '{{SEVERITY}}': severity,
            '{{VULN_COUNT}}': vuln_count,
            '{{ML_PROB}}': ml_prob,
            '{{LLM_OVERALL}}': llm_overall,
            '{{CONFLICT_DETECTED}}': conflict_detected,
            '{{CONFLICT_REASON}}': conflict_reason,
            '{{BUSINESS_DOMAIN}}': meta.get('business_domain', 'N/A'),
            '{{CHAIN}}': meta.get('chain', 'N/A'),
            '{{COMPILER_VERSION}}': meta.get('compiler_version', 'N/A'),
            '{{NOTES}}': meta.get('notes', 'N/A'),
            '{{UPLOAD_TIME}}': meta.get('upload_time', 'N/A'),
            '{{TOP_FINDINGS_HTML}}': top_findings_html,
        }
        for k, v in replacements.items():
            html = html.replace(k, str(v))

        return html
    except Exception as e:
        app.logger.error(f"报告模板渲染失败: {e}")
        # 回退：返回最小可读信息，避免整体失败
        return f"<html><body><h3>报告生成失败</h3><p>{e}</p></body></html>"

def generate_pdf_report(html_content, pdf_path):
    """
    将HTML内容转换为PDF文件
    
    Args:
        html_content: HTML内容字符串
        pdf_path: 输出PDF文件路径
    
    Returns:
        bool: 是否成功生成PDF
    """
    try:
        # 优先使用 weasyprint（推荐，支持CSS3）
        return generate_pdf_with_weasyprint(html_content, pdf_path)
    except Exception as e:
        app.logger.error(f"weasyprint PDF生成失败: {e}")
        try:
            # 备选方案：使用 pdfkit
            return generate_pdf_with_pdfkit(html_content, pdf_path)
        except Exception as e2:
            app.logger.error(f"pdfkit PDF生成也失败: {e2}")
            return False

def generate_pdf_with_weasyprint(html_content, pdf_path):
    """使用 weasyprint 生成PDF"""
    try:
        # 创建HTML对象
        html = HTML(string=html_content)

        # 从静态文件加载CSS（职责分离）
        css_path = os.path.join('static', 'report.css')
        css = CSS(filename=css_path)

        # 生成PDF
        html.write_pdf(pdf_path, stylesheets=[css])
        
        app.logger.info(f"使用 weasyprint 成功生成PDF: {pdf_path}")
        return True
        
    except Exception as e:
        app.logger.error(f"weasyprint PDF生成失败: {e}")
        return False

def generate_pdf_with_pdfkit(html_content, pdf_path):
    """使用 pdfkit 生成PDF"""
    try:
        # pdfkit 配置选项
        options = {
            'page-size': 'A4',
            'margin-top': '1cm',
            'margin-right': '1cm',
            'margin-bottom': '1cm',
            'margin-left': '1cm',
            'encoding': 'UTF-8',
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        # 生成PDF
        pdfkit.from_string(html_content, pdf_path, options=options)
        
        app.logger.info(f"使用 pdfkit 成功生成PDF: {pdf_path}")
        return True
        
    except Exception as e:
        app.logger.error(f"pdfkit PDF生成失败: {e}")
        return False

def generate_findings_html(findings):
    """生成发现问题的HTML"""
    if not findings:
        return "<p>未发现明显的安全问题。</p>"
    
    html = ""
    for i, finding in enumerate(findings, 1):
        html += f"""
        <div class="finding">
            <div class="finding-header">问题 {i}: {finding.get('vuln_type', 'Unknown')}</div>
            <div class="finding-details">
                <p><strong>位置:</strong> {finding.get('location', 'N/A')}</p>
                <p><strong>严重性:</strong> {finding.get('severity', 'Low')}</p>
                <p><strong>置信度:</strong> {finding.get('confidence', 'N/A')}</p>
                <p><strong>证据:</strong> {finding.get('evidence', 'N/A')}</p>
                <p><strong>建议:</strong> {finding.get('suggestion', 'N/A')}</p>
            </div>
        </div>
        """
    
    return html

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

# ================= 新增的作业管理API端点 =====================

@app.route('/api/jobs', methods=['POST'])
def create_job():
    """
    创建分析作业
    参数：upload_id, slice_kind, llm_model, notes
    """
    try:
        data = request.get_json()
        if not data or not data.get('upload_id'):
            return jsonify({'error': '缺少必需参数 upload_id'}), 400
        
        upload_id = data['upload_id']
        
        # 验证upload_id是否存在
        upload_dir = os.path.join(app.config['UPLOADS_FOLDER'], upload_id)
        if not os.path.exists(upload_dir):
            return jsonify({'error': f'上传ID {upload_id} 不存在'}), 404
        
        # 获取当前用户（可选）
        user = get_current_user()
        
        # 创建作业记录
        job = Job(
            upload_id=upload_id,
            user_id=user.id if user else None,
            slice_kind=data.get('slice_kind', 'full'),
            llm_model=data.get('llm_model', 'deepseek'),
            notes=data.get('notes', '')
        )
        
        db.session.add(job)
        db.session.commit()
        
        # 启动后台处理线程
        job_thread = Thread(
            target=process_job_workflow, 
            args=(job.id,), 
            name=f"JobThread-{job.id[:8]}"
        )
        job_thread.daemon = True
        job_thread.start()
        
        app.logger.info(f"作业 {job.id} 已创建并启动处理")
        
        return jsonify({
            'job_id': job.id,
            'message': '作业创建成功，正在后台处理'
        }), 201
        
    except Exception as e:
        app.logger.error(f"创建作业失败: {e}")
        return jsonify({'error': f'创建作业失败: {str(e)}'}), 500

@app.route('/api/jobs/<string:job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    查询作业状态与阶段产物
    """
    try:
        job = Job.query.get(job_id)
        if not job:
            return jsonify({'error': '作业不存在'}), 404
        
        # 权限检查（如果有用户系统）
        user = get_current_user()
        if job.user_id and user and job.user_id != user.id:
            return jsonify({'error': '无权访问此作业'}), 403
        
        # 构建作业状态响应
        duration = None
        if job.started_at and job.finished_at:
            duration = f"{round((job.finished_at - job.started_at).total_seconds(), 2)}s"
        elif job.started_at:
            duration = f"{round((datetime.utcnow() - job.started_at).total_seconds(), 2)}s (运行中)"
        
        response = {
            'job_id': job.id,
            'upload_id': job.upload_id,
            'status': job.status,
            'status_description': JOB_STATES.get(job.status, job.status),
            'current_stage': job.current_stage,
            'progress': round(job.progress, 3),
            'stage_progress': round(job.stage_progress, 3),
            'error_msg': job.error_msg,
            'slice_kind': job.slice_kind,
            'llm_model': job.llm_model,
            'notes': job.notes,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'finished_at': job.finished_at.isoformat() if job.finished_at else None,
            'duration': duration,
            'final_score': job.final_score,
            'severity': job.severity,
            'vulnerability_count': job.vulnerability_count,
            'artifacts': {
                'source_path': job.source_path,
                'preproc_path': job.preproc_path,
                'llm_result_path': job.llm_result_path,
                'ml_result_path': job.ml_result_path,
                'fusion_result_path': job.fusion_result_path,
                'html_report_path': job.html_report_path,
                'pdf_report_path': job.pdf_report_path
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f"查询作业状态失败: {e}")
        return jsonify({'error': f'查询失败: {str(e)}'}), 500

@app.route('/api/reports/<string:job_id>.html', methods=['GET'])
def get_html_report(job_id):
    """
    查看HTML报告
    """
    try:
        job = Job.query.get(job_id)
        if not job:
            return jsonify({'error': '作业不存在'}), 404
        
        if job.status != 'REPORT_READY' or not job.html_report_path:
            return jsonify({'error': '报告尚未生成'}), 404
        
        # 权限检查
        user = get_current_user()
        if job.user_id and user and job.user_id != user.id:
            return jsonify({'error': '无权访问此报告'}), 403
        
        # 读取HTML报告文件
        upload_dir = os.path.join(app.config['UPLOADS_FOLDER'], job.upload_id)
        report_path = os.path.join(upload_dir, job.html_report_path)
        
        if not os.path.exists(report_path):
            return jsonify({'error': '报告文件不存在'}), 404
        
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}
        
    except Exception as e:
        app.logger.error(f"获取HTML报告失败: {e}")
        return jsonify({'error': f'获取报告失败: {str(e)}'}), 500

@app.route('/api/reports/<string:job_id>.pdf', methods=['GET'])
def download_pdf_report(job_id):
    """
    下载PDF报告
    """
    try:
        job = Job.query.get(job_id)
        if not job:
            return jsonify({'error': '作业不存在'}), 404
        
        if job.status != 'REPORT_READY' or not job.pdf_report_path:
            return jsonify({'error': 'PDF报告尚未生成'}), 404
        
        # 权限检查
        user = get_current_user()
        if job.user_id and user and job.user_id != user.id:
            return jsonify({'error': '无权访问此报告'}), 403
        
        # 返回PDF文件
        upload_dir = os.path.join(app.config['UPLOADS_FOLDER'], job.upload_id)
        return send_from_directory(
            upload_dir, 
            job.pdf_report_path, 
            as_attachment=True,
            download_name=f'contract_analysis_{job_id[:8]}.pdf'
        )
        
    except Exception as e:
        app.logger.error(f"下载PDF报告失败: {e}")
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """
    返回可选配置信息
    """
    try:
        config = {
            'slice_kinds': [
                {'value': 'full', 'label': '完整合约', 'description': '分析整个合约文件'}
            ],
            'llm_models': [
                {'value': 'deepseek', 'label': 'DeepSeek', 'description': 'DeepSeek 大语言模型'}
            ],
            'business_domains': [
                {'value': 'cross_border', 'label': '跨境贸易', 'description': '跨境贸易合规与安全审计'}
            ],
            'chains': [
                {'value': 'ethereum', 'label': 'Ethereum', 'description': '以太坊主网'},
                {'value': 'bsc', 'label': 'BSC', 'description': '币安智能链'},
                {'value': 'polygon', 'label': 'Polygon', 'description': 'Polygon 网络'},
                {'value': 'arbitrum', 'label': 'Arbitrum', 'description': 'Arbitrum Layer2'},
                {'value': 'optimism', 'label': 'Optimism', 'description': 'Optimism Layer2'}
            ],
            'compiler_versions': [
                {'value': '0.8.19', 'label': 'Solidity 0.8.19'},
                {'value': '0.8.18', 'label': 'Solidity 0.8.18'},
                {'value': '0.8.17', 'label': 'Solidity 0.8.17'},
                {'value': 'auto', 'label': '自动检测'}
            ],
            'system_info': {
                'version': 'SCANA v2.0',
                'features': ['LLM分析', 'ML推理', '结果融合', '智能报告'],
                'supported_formats': ['.sol'],
                'max_file_size': '1MB',
                'processing_timeout': '300s'
            }
        }
        
        return jsonify(config), 200
        
    except Exception as e:
        app.logger.error(f"获取配置失败: {e}")
        return jsonify({'error': f'获取配置失败: {str(e)}'}), 500

@app.route('/api/jobs/list', methods=['GET'])
def list_jobs():
    """
    获取作业列表（支持分页和筛选）
    """
    try:
        # 获取查询参数
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)  # 限制最大20条
        status = request.args.get('status')
        
        # 构建查询
        query = Job.query
        
        # 用户权限过滤
        user = get_current_user()
        if user:
            query = query.filter_by(user_id=user.id)
        else:
            # 匿名用户只能看到没有关联用户的作业
            query = query.filter_by(user_id=None)
        
        # 状态过滤
        if status and status in JOB_STATES:
            query = query.filter_by(status=status)
        
        # 分页查询
        jobs = query.order_by(Job.created_at.desc()).paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        # 构建响应
        job_list = []
        for job in jobs.items:
            duration = None
            if job.started_at and job.finished_at:
                duration = f"{round((job.finished_at - job.started_at).total_seconds(), 2)}s"
            elif job.started_at:
                duration = f"{round((datetime.utcnow() - job.started_at).total_seconds(), 2)}s (运行中)"
            
            job_info = {
                'job_id': job.id,
                'upload_id': job.upload_id,
                'status': job.status,
                'status_description': JOB_STATES.get(job.status, job.status),
                'progress': round(job.progress, 3),
                'final_score': job.final_score,
                'severity': job.severity,
                'vulnerability_count': job.vulnerability_count,
                'created_at': job.created_at.isoformat(),
                'duration': duration,
                'slice_kind': job.slice_kind,
                'llm_model': job.llm_model
            }
            job_list.append(job_info)
        
        response = {
            'jobs': job_list,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': jobs.total,
                'pages': jobs.pages,
                'has_next': jobs.has_next,
                'has_prev': jobs.has_prev
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f"获取作业列表失败: {e}")
        return jsonify({'error': f'获取作业列表失败: {str(e)}'}), 500

@app.route('/api/uploads', methods=['POST'])
def upload_contract():
    """
    合约文件上传接口：
    - 支持表单上传（multipart/form-data），字段：file, business_domain, chain, compiler_version, notes
    - 生成唯一 upload_id
    - 保存文件到 uploads/{upload_id}/
    - 保存元信息到 uploads/{upload_id}/meta.json
    - 返回 upload_id
    """
    # 检查文件
    if 'file' not in request.files:
        return jsonify({'error': '未检测到上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    # 生成唯一 upload_id
    upload_id = str(uuid.uuid4())
    upload_dir = os.path.join(app.config['UPLOADS_FOLDER'], upload_id)
    os.makedirs(upload_dir, exist_ok=True)
    # 保存文件
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)
    # 收集元信息
    meta = {
        'upload_id': upload_id,
        'filename': file.filename,

        'business_domain': 'cross_border',
        'chain': request.form.get('chain', ''),
        'compiler_version': request.form.get('compiler_version', ''),
        'notes': request.form.get('notes', ''),
        'upload_time': datetime.utcnow().isoformat()
    }
    # 保存元信息到 meta.json
    with open(os.path.join(upload_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # 返回 upload_id
    return jsonify({'upload_id': upload_id}), 200


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