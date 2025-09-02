# 合约安全分析后端（跨境贸易专用）

跨境贸易场景（cross_border）下的 Solidity 智能合约安全分析后端。采用 DeepSeek LLM + 本地 ML 模型（BLSTM+Attention）双引擎，自动融合并生成 HTML/PDF 报告。

## 快速开始（开发者）

1) 创建与激活环境
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) 准备模型文件（必需）
```
model/w2v/checkpoints/blstm_epoch470.pt
model/w2v/checkpoints/sg_epoch1700.pt
```

3) 配置环境变量（二选一）
- 方式A：使用 .env（推荐，在 app.py 顶部可 load_dotenv()）
```bash
# 必需
LLM_API_KEY=your_deepseek_api_key
SECRET_KEY=your_secret_key_here
# 可选
LLM_API_ENDPOINT=https://api.deepseek.com/v1/chat/completions
LLM_TIMEOUT=60
LLM_MAX_RETRY=3
LLM_MAX_TOKENS=2048
DATABASE_URL=sqlite:///scana.db
```
- 方式B：直接 export
```bash
export LLM_API_KEY=your_deepseek_api_key
export SECRET_KEY=your_secret_key_here
```

4) 启动
```bash
python app.py
# 服务: http://localhost:8000
```

## 依赖
- Python 包：Flask, SQLAlchemy, Flask-CORS, requests, torch, weasyprint, pdfkit
- 系统库（二选一）
  - weasyprint：cairo, pango, gdk-pixbuf, libffi
  - pdfkit：wkhtmltopdf

## 核心端点
- 上传合约（自动触发 LLM 初判）
  - POST `/api/uploads` (multipart/form-data: file, chain, compiler_version, notes)
  - 说明：业务域固定为跨境贸易（cross_border），无需传 business_domain
- 创建作业（后台执行完整流程）
  - POST `/api/jobs`  {"upload_id": "..."}
- 作业详情/状态
  - GET `/api/jobs/{job_id}`
- 作业列表
  - GET `/api/jobs/list?status=...&page=1&per_page=20`
- 报告查看/下载
  - GET `/api/reports/{job_id}.html`
  - GET `/api/reports/{job_id}.pdf`
- 配置查询（仅跨境贸易）
  - GET `/api/config`  （business_domains 仅 `cross_border`）

## 产物结构（uploads/{upload_id}/）
- `meta.json`：上传元信息（filename、chain、compiler_version、notes、upload_time）
- 源文件（原始 .sol）
- `preproc/`：预处理副本
- `llm_result.json`：LLM 初判
- `ml_result.json`：ML 推理
- `fusion_result.json`：融合结果（final_score、severity、top_findings 等）
- `report.html`：最终报告（模板：`static/report_template.html`）
- `report.pdf`：PDF 报告（weasyprint 优先，pdfkit 兜底）

## 处理流程（跨境贸易-only）
1. 上传合约（自动 LLM 初判保存为 llm_result.json）
2. 创建作业：PREPROCESSING → LLM_ANALYZING → ML_INFER → FUSING → REPORT_READY
3. 融合：final_score = α·ML_prob + β·LLM_overall + γ·issue_aggregation（默认 α=0.4, β=0.4, γ=0.2）
4. 报告：基于 `static/report_template.html` 渲染 HTML，再生成 PDF

## 最小验证
```bash
# 1) 上传
curl -X POST http://localhost:8000/api/uploads \
  -F "file=@contract.sol" \
  -F "chain=ethereum" \
  -F "compiler_version=0.8.19" \
  -F "notes=跨境贸易结算合约"
# ⇒ {"upload_id":"..."}

# 2) 创建作业
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"upload_id":"..."}'
# ⇒ {"job_id":"..."}

# 3) 轮询状态
curl http://localhost:8000/api/jobs/{job_id}

# 4) 查看/下载报告
curl http://localhost:8000/api/reports/{job_id}.html
curl -O http://localhost:8000/api/reports/{job_id}.pdf
```

## 注意
- LLM_API_KEY 未配置时，LLM 初判会失败（流程仍可继续，但融合与报告信息受限）
- 模板为浅色专业风，报告页头提供“下载PDF”按钮
- 仅支持跨境贸易业务域（cross_border），其他业务域不在本项目范围内

