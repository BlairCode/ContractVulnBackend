# 智能合约安全分析系统 API 使用指南

## 概述

本系统提供完整的智能合约安全分析能力，集成了 DeepSeek LLM 和机器学习模型，支持自动化的漏洞检测、结果融合和报告生成。

## 系统架构

```
用户上传合约 → LLM初步分析 → 创建作业 → ML推理 → 结果融合 → 生成报告
```

## API 端点

### 1. 文件上传

**POST /api/uploads**

上传智能合约文件并自动进行 LLM 初步分析。

**请求格式：** multipart/form-data

**参数：**
- `file` (必需): 合约文件 (.sol)
- `business_domain`: 业务域 (defi/nft/dao/gaming/cross_border/other)
- `chain`: 区块链网络 (ethereum/bsc/polygon/arbitrum/optimism/other)
- `compiler_version`: 编译器版本 (0.8.19/0.8.18/auto等)
- `notes`: 备注信息

**响应：**
```json
{
  "upload_id": "uuid-string"
}
```

**示例：**
```bash
curl -X POST http://localhost:8000/api/uploads \
  -F "file=@contract.sol" \
  -F "business_domain=defi" \
  -F "chain=ethereum" \
  -F "compiler_version=0.8.19" \
  -F "notes=测试合约"
```

### 2. 创建分析作业

**POST /api/jobs**

基于上传的文件创建完整的分析作业。

**请求格式：** application/json

**参数：**
```json
{
  "upload_id": "uuid-string",
  "slice_kind": "full|function|statement",
  "llm_model": "deepseek|gpt-4|claude",
  "notes": "作业备注"
}
```

**响应：**
```json
{
  "job_id": "uuid-string",
  "message": "作业创建成功，正在后台处理"
}
```

### 3. 查询作业状态

**GET /api/jobs/{job_id}**

查询作业的详细状态和进度。

**响应：**
```json
{
  "job_id": "uuid-string",
  "upload_id": "uuid-string",
  "status": "PENDING|PREPROCESSING|LLM_ANALYZING|ML_INFER|FUSING|REPORT_READY|FAILED",
  "status_description": "状态描述",
  "current_stage": "当前阶段",
  "progress": 0.85,
  "stage_progress": 0.3,
  "error_msg": null,
  "final_score": 0.75,
  "severity": "High",
  "vulnerability_count": 3,
  "duration": "45.2s",
  "artifacts": {
    "llm_result_path": "llm_result.json",
    "ml_result_path": "ml_result.json",
    "fusion_result_path": "fusion_result.json",
    "html_report_path": "report.html"
  }
}
```

### 4. 获取作业列表

**GET /api/jobs/list**

获取作业历史列表，支持分页和筛选。

**查询参数：**
- `page`: 页码 (默认1)
- `per_page`: 每页数量 (默认20，最大100)
- `status`: 状态筛选

**响应：**
```json
{
  "jobs": [
    {
      "job_id": "uuid-string",
      "status": "REPORT_READY",
      "progress": 1.0,
      "final_score": 0.75,
      "severity": "High",
      "vulnerability_count": 3,
      "created_at": "2024-01-01T12:00:00",
      "duration": "45.2s"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 50,
    "pages": 3,
    "has_next": true,
    "has_prev": false
  }
}
```

### 5. 查看 HTML 报告

**GET /api/reports/{job_id}.html**

在线查看分析报告。

**响应：** HTML 页面

### 6. 下载 PDF 报告

**GET /api/reports/{job_id}.pdf**

下载 PDF 格式的分析报告。

**响应：** PDF 文件下载

### 7. 获取系统配置

**GET /api/config**

获取系统支持的配置选项。

**响应：**
```json
{
  "slice_kinds": [
    {"value": "full", "label": "完整合约", "description": "分析整个合约文件"}
  ],
  "llm_models": [
    {"value": "deepseek", "label": "DeepSeek", "description": "DeepSeek 大语言模型"}
  ],
  "business_domains": [...],
  "chains": [...],
  "compiler_versions": [...],
  "system_info": {
    "version": "SCANA v2.0",
    "features": ["LLM分析", "ML推理", "结果融合", "智能报告"]
  }
}
```

## 工作流程

### 完整分析流程

1. **上传文件**
   ```bash
   curl -X POST http://localhost:8000/api/uploads -F "file=@contract.sol"
   # 返回: {"upload_id": "abc-123"}
   ```

2. **创建作业**
   ```bash
   curl -X POST http://localhost:8000/api/jobs \
     -H "Content-Type: application/json" \
     -d '{"upload_id": "abc-123", "slice_kind": "full", "llm_model": "deepseek"}'
   # 返回: {"job_id": "def-456"}
   ```

3. **轮询状态**
   ```bash
   curl http://localhost:8000/api/jobs/def-456
   # 状态变化: PENDING → PREPROCESSING → LLM_ANALYZING → ML_INFER → FUSING → REPORT_READY
   ```

4. **查看报告**
   ```bash
   # 在线查看
   curl http://localhost:8000/api/reports/def-456.html
   
   # 下载PDF
   curl -O http://localhost:8000/api/reports/def-456.pdf
   ```

## 状态机说明

```
PENDING      → 等待开始
PREPROCESSING → 预处理中 (代码解析、切片等)
LLM_ANALYZING → LLM分析中 (DeepSeek 分析)
ML_INFER     → ML推理中 (机器学习模型预测)
FUSING       → 结果融合中 (LLM + ML 结果融合)
REPORT_READY → 报告已生成 (完成)
FAILED       → 处理失败
```

## 融合算法

系统采用加权融合算法：

```
final_score = α × ML_probability + β × LLM_overall_score + γ × issue_aggregation
```

默认权重：
- α = 0.4 (ML概率权重)
- β = 0.4 (LLM整体评分权重)  
- γ = 0.2 (问题聚合权重)

严重性映射：
- Critical: ≥ 0.8
- High: ≥ 0.6  
- Medium: ≥ 0.4
- Low: < 0.4

## 环境配置

创建 `.env` 文件：

```bash
# LLM 配置
LLM_API_KEY=your_deepseek_api_key
LLM_API_ENDPOINT=https://api.deepseek.com/v1/chat/completions
LLM_TIMEOUT=60
LLM_MAX_RETRY=3
LLM_MAX_TOKENS=2048

# 数据库
DATABASE_URL=sqlite:///scana.db

# Flask
SECRET_KEY=your_secret_key_here
```

## 错误处理

所有 API 都遵循统一的错误响应格式：

```json
{
  "error": "错误描述信息"
}
```

常见错误码：
- 400: 请求参数错误
- 401: 认证失败  
- 403: 权限不足
- 404: 资源不存在
- 500: 服务器内部错误

## 性能说明

- 文件上传后立即返回 upload_id
- LLM 分析在上传时自动触发（异步）
- 完整作业处理时间通常为 30-120 秒
- 支持并发处理多个作业
- 所有产物持久化存储，可重复访问

