# 开发者安装指南

## 快速开始

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd ContractVulnBackend
```

### 2. 创建虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或者
venv\Scripts\activate     # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量
复制 `.env.example` 为 `.env` 并编辑：
```bash
cp .env.example .env
# 编辑 .env 文件，设置你的 API Key
```

### 5. 运行应用
```bash
python app.py
```

应用将在 http://localhost:8000 启动

## 依赖说明

### 核心依赖
- Flask 2.3.3+
- PyTorch 1.9.0+
- SQLAlchemy 3.0.5+

### PDF生成
- weasyprint (主要)
- pdfkit (备选)

### 系统依赖
- Python 3.8+
- 如果使用 weasyprint，需要安装 cairo、pango 等

## 开发环境

### 目录结构
```
ContractVulnBackend/
├── app.py              # 主应用文件
├── requirements.txt    # Python依赖
├── uploads/           # 上传文件存储
├── logs/              # 日志文件
├── static/            # 静态文件
└── .env               # 环境配置
```

### 环境变量
```bash
# LLM配置
LLM_API_KEY=your_deepseek_api_key
LLM_API_ENDPOINT=https://api.deepseek.com/v1/chat/completions

# 数据库
DATABASE_URL=sqlite:///scana.db

# Flask
SECRET_KEY=your_secret_key
```

## 测试

### 1. 检查依赖
```bash
python -c "
import flask, torch, requests
print('✅ 核心依赖检查通过')
"
```

### 2. 测试API
```bash
# 启动应用后
curl http://localhost:8000/api/config
```

## 常见问题

### PDF生成失败
```bash
# 安装PDF生成依赖
pip install weasyprint pdfkit

# macOS
brew install cairo pango gdk-pixbuf libffi wkhtmltopdf

# Ubuntu
sudo apt-get install libcairo2 libpango-1.0-0 libpangocairo-1.0-0 wkhtmltopdf
```

### 模型文件缺失
确保 `model/` 目录下有必要的模型文件：
- `model/w2v/checkpoints/blstm_epoch470.pt`
- `model/w2v/checkpoints/sg_epoch1700.pt`

## 开发建议

1. 使用虚拟环境隔离依赖
2. 定期更新 requirements.txt
3. 查看 logs/ 目录了解运行状态
4. 使用 API_USAGE.md 了解接口用法
