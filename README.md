# 智能合约安全分析系统 (SCANA)

一个集成了大语言模型(LLM)和机器学习的智能合约安全分析平台，支持自动化的漏洞检测、结果融合和报告生成。

## 🚀 核心特性

- **双重分析引擎**: DeepSeek LLM + 机器学习模型
- **智能结果融合**: 加权融合算法，冲突检测与解决
- **完整工作流**: 上传 → 分析 → 融合 → 报告
- **多格式报告**: HTML + PDF 双格式输出
- **RESTful API**: 完整的后端接口
- **状态管理**: 实时进度跟踪和状态监控

## 🏗️ 系统架构

```
用户上传合约 → LLM初步分析 → 创建作业 → ML推理 → 结果融合 → 生成报告
```

### 状态机流程
```
PENDING → PREPROCESSING → LLM_ANALYZING → ML_INFER → FUSING → REPORT_READY
```

## 📁 项目结构

```
ContractVulnBackend/
├── app.py                 # 主应用文件（包含所有后端逻辑）
├── requirements.txt       # Python依赖
├── README.md             # 项目说明
├── API_USAGE.md          # API使用指南
├── DEVELOPER_SETUP.md    # 开发者安装指南
├── .env                  # 环境配置文件（包含完整模板）
├── uploads/              # 文件存储目录（自动创建）
├── logs/                 # 日志文件目录（自动创建）
└── static/               # 前端静态文件
```

## 🛠️ 技术栈

- **后端框架**: Flask + SQLAlchemy
- **机器学习**: PyTorch + 自定义BLSTM+Attention模型
- **大语言模型**: DeepSeek API (支持环境变量配置)
- **数据库**: SQLite (支持PostgreSQL，通过环境变量配置)
- **PDF生成**: WeasyPrint (推荐) + pdfkit (备选)
- **异步处理**: 多线程后台任务
- **文件存储**: 本地文件系统，结构化目录管理

## 🚀 快速开始

### 1. 环境要求
- Python 3.8+
- 有效的 DeepSeek API Key

### 2. 安装依赖
```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境
```bash
# 编辑 .env 文件，设置你的 DeepSeek API Key
# 主要需要修改的配置项：
# LLM_API_KEY=your_deepseek_api_key_here
# SECRET_KEY=your_secret_key_here_change_in_production
```

### 4. 运行应用
```bash
python app.py
```

应用将在 http://localhost:8000 启动

**注意**：首次运行前，请确保：
1. 已编辑 `.env` 文件并设置正确的 DeepSeek API Key
2. 已修改 `.env` 文件中的 SECRET_KEY
3. `model/` 目录下有必要的模型文件


**快速配置检查**：
```bash
# 检查.env文件是否存在
ls -la .env

# 检查关键配置项
grep "LLM_API_KEY\|SECRET_KEY" .env
```

## 📚 API 接口

### 核心接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/uploads` | POST | 上传合约文件 |
| `/api/jobs` | POST | 创建分析作业 |
| `/api/jobs/{job_id}` | GET | 查询作业状态 |
| `/api/jobs/list` | GET | 获取作业列表 |
| `/api/reports/{job_id}.html` | GET | 查看HTML报告 |
| `/api/reports/{job_id}.pdf` | GET | 下载PDF报告 |
| `/api/config` | GET | 获取系统配置 |

### 使用示例

```bash
# 1. 上传合约
curl -X POST http://localhost:8000/api/uploads \
  -F "file=@contract.sol" \
  -F "business_domain=defi"

# 2. 创建作业
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"upload_id": "abc-123", "slice_kind": "full"}'

# 3. 查询状态
curl http://localhost:8000/api/jobs/def-456

# 4. 查看报告
curl http://localhost:8000/api/reports/def-456.html
```

## 🔬 融合算法

系统采用加权融合算法：

```
final_score = α × ML_probability + β × LLM_overall_score + γ × issue_aggregation
```

**默认权重配置**:
- α = 0.4 (ML概率权重)
- β = 0.4 (LLM整体评分权重)  
- γ = 0.2 (问题聚合权重)

**严重性映射**:
- Critical: ≥ 0.8
- High: ≥ 0.6  
- Medium: ≥ 0.4
- Low: < 0.4

## 📊 报告特性

### HTML报告
- 响应式设计，支持移动端
- 实时数据展示
- 交互式图表和进度条

### PDF报告
- 专业格式，适合打印
- 分页优化，避免内容截断
- 页眉页脚，包含页码信息

## 🔧 配置选项

### 环境变量配置
系统支持通过 `.env` 文件或环境变量进行配置。项目已包含完整的 `.env` 配置文件模板。

**主要配置项说明**：
- `LLM_API_KEY`: DeepSeek API密钥（必须设置）
- `LLM_API_ENDPOINT`: API端点（默认已设置）
- `LLM_TIMEOUT`: 请求超时时间（默认60秒）
- `LLM_MAX_RETRY`: 最大重试次数（默认3次）
- `LLM_MAX_TOKENS`: 最大token数（默认2048）
- `DATABASE_URL`: 数据库连接（默认SQLite）
- `SECRET_KEY`: Flask密钥（建议修改）

### 支持的配置项
- **LLM配置**：API Key、Endpoint、超时、重试、最大token
- **数据库配置**：支持SQLite和PostgreSQL
- **Flask配置**：密钥、端口等

### 配置文件示例
`.env` 文件包含以下配置模板：
```bash
# LLM配置
LLM_API_KEY=your_deepseek_api_key_here
LLM_API_ENDPOINT=https://api.deepseek.com/v1/chat/completions
LLM_TIMEOUT=60
LLM_MAX_RETRY=3
LLM_MAX_TOKENS=2048

# 数据库配置
DATABASE_URL=sqlite:///scana.db

# Flask配置
SECRET_KEY=your_secret_key_here_change_in_production
```

### 硬编码配置
以下配置在代码中硬编码，暂不支持环境变量覆盖：
- `UPLOADS_FOLDER=uploads`：上传文件存储目录
- `LOGS_FOLDER=logs`：日志文件目录
- `CHECKPOINT_PATH`：ML模型检查点路径
- `W2V_CP_PATH`：词向量模型路径
- `VULNERABILITY_THRESHOLD=0.2`：漏洞预测阈值

### 业务配置
- 支持多种业务域：DeFi、NFT、DAO、游戏、跨境贸易等（通过API配置）
- 支持多种区块链：Ethereum、BSC、Polygon、Arbitrum等（通过API配置）
- 支持多种编译器版本：0.6.x - 0.8.x（通过API配置）
- 支持多种切片类型：完整合约、函数级别、语句级别

## 🚨 安全特性

- 敏感信息清洗（预留钩子）
- 文件大小限制
- API调用频率限制
- 用户权限隔离
- 详细审计日志

## 📈 性能说明

- 文件上传后立即返回ID
- LLM分析异步执行
- 完整作业处理时间：30-120秒
- 支持并发处理
- 产物持久化存储

## 🐛 故障排除

### 常见问题

1. **PDF生成失败**
   ```bash
   # 安装PDF生成依赖
   pip install weasyprint pdfkit
   
   # macOS
   brew install cairo pango gdk-pixbuf libffi wkhtmltopdf
   
   # Ubuntu
   sudo apt-get install libcairo2 libpango-1.0-0 libpangocairo-1.0-0 wkhtmltopdf
   ```

2. **模型文件缺失**
   - 确保 `model/` 目录下有必要的模型文件：
     - `model/w2v/checkpoints/blstm_epoch470.pt`
     - `model/w2v/checkpoints/sg_epoch1700.pt`
   - 如果没有模型文件，系统会启动失败

3. **API调用失败**
   - 检查 `.env` 文件中的API Key配置
   - 确认网络连接正常

### 日志查看
```bash
# 查看应用日志
tail -f logs/scana_*.log
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 支持

如有问题或建议，请：
- 提交 Issue
- 发送邮件
- 查看 [API_USAGE.md](API_USAGE.md) 获取详细使用说明

## ⚠️ 重要说明

### 配置文件
- 项目已包含完整的 `.env` 配置文件模板
- 必须设置有效的 DeepSeek API Key
- 建议修改 SECRET_KEY 为随机字符串
- 首次运行会自动创建必要的目录结构

### 模型文件
- 需要预训练的机器学习模型文件
- 模型文件路径在代码中硬编码
- 没有模型文件系统无法启动

### 依赖安装
- 推荐使用虚拟环境
- PDF生成需要额外的系统依赖
- 详细安装说明请参考 [DEVELOPER_SETUP.md](DEVELOPER_SETUP.md)

---

**开始使用**: 查看 [DEVELOPER_SETUP.md](DEVELOPER_SETUP.md) 获取开发者安装指南
