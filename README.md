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
  - POST `/api/reports/sample` （生成跨境贸易示例报告，返回路径）

## 产物结构（uploads/{upload_id}/）
- `meta.json`：上传元信息（filename、chain、compiler_version、notes、upload_time）
- 源文件（原始 .sol）
- `preproc/`：预处理副本
- `llm_result.json`：LLM 初判
- `ml_result.json`：ML 推理
- `fusion_result.json`：融合结果（final_score、severity、top_findings 等）
- `report.html`：最终报告（模板：`static/report_template.html`）
- `report.pdf`：PDF 报告（WeasyPrint 优先，pdfkit 兜底；样式：`static/report.css`）

## 处理流程（跨境贸易场景）
1. 上传合约：自动触发 LLM 初判，保存 `llm_result.json`
2. 创建作业：PREPROCESSING → LLM_ANALYZING（多阶段：precheck → compliance → deep_audit，聚合结果） → ML_INFER → FUSING → REPORT_READY
3. 融合（固定跨境贸易权重）：final_score = α·ML_prob + β·LLM_overall + γ·issue_aggregation（α=0.35, β=0.45, γ=0.20；冲突阈=0.28）
4. 报告：基于 `static/report_template.html` 渲染 HTML，使用 `static/report.css` 生成 PDF（WeasyPrint 优先，pdfkit 兜底）

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

# 5) 生成示例报告（跨境贸易演示）
curl -X POST http://localhost:8000/api/reports/sample
```

## 注意
- LLM_API_KEY 未配置时，LLM 初判会失败（流程仍可继续，但融合与报告信息受限）
