# 智能合约漏洞检测后端接口

## 项目简介
这个是一个智能合约漏洞检测的后端接口，用 Flask 搭建，集成了 PyTorch 的 AI 模型（SkipGram 和 Fusion Model）来分析 `.sol` 文件，检测重入漏洞（Reentrancy）。目前完成了文件上传和漏洞检测的 API，制作有简易前端示例。

## 环境要求
- Python 3.8 或以上
- 依赖库列在 `requirements.txt` 中
- 需要模型文件：
  - `model/w2v/sg_epoch0.pt`（SkipGram 模型）
  - `model/fusion_model.pt`（分类模型）

## 安装步骤
1. 确保项目文件夹包含 `app.py` 和 `model/` 目录。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   - Mac M1/M2 用户：PyTorch 要单独装支持 MPS 的版本：
     ```bash
     pip install torch torchvision
     ```
     参考：https://pytorch.org/get-started/locally/

## 配置
1. 复制 `.env.example` 到 `.env`：
   ```bash
   cp .env.example .env
   ```
2. 编辑 `.env`，设置路径和端口：
   ```
   UPLOAD_FOLDER=./uploads
   MODEL_CHECKPOINT=./model/w2v/sg_epoch0.pt
   FUSION_CHECKPOINT=./model/fusion_model.pt
   PORT=8888
   HOST=0.0.0.0
   LOG_FILE=./app.log
   ```
   - Windows 用反斜杠（`\`），Mac 用正斜杠（`/`）。

## 运行
1. 启动后端：
   ```bash
   python app.py
   ```
   - 默认跑在 `http://localhost:8888`，端口被占会自动试下一个（最多5次）。
   - 可改端口：
     ```bash
     export PORT=9999  # Mac/Linux
     set PORT=9999     # Windows
     python app.py
     ```

## 接口说明
- **`GET /`**：返回前端页面（目前是 `index.html`，但还没完全实现）。
- **`POST /upload`**：上传 `.sol` 文件，返回文件路径。
  - 请求：带 `file` 字段的 FormData。
  - 响应：`{"message": "文件上传成功", "filepath": "..."}`。
- **`POST /detect`**：检测漏洞，返回结果。
  - 请求：JSON 格式，`{"filepath": "...", "vulnerabilities": ["Reentrancy"]}`。
  - 响应：`{"filepath": "...", "results": {"predicted_class": 0/1, "probabilities": [...]}}`。

## 当前状态
- 已完成：
  - 文件上传和保存功能。
  - AI 模型加载和漏洞检测逻辑。
  - 基本的错误处理和日志记录。
- 未完成：
  - 前端界面（`index.html` 只是占位，需要完善）。
  - 可能需要更多漏洞类型支持（目前只检测 Reentrancy）。

## 下一步建议
- 完善前端，连接 `/upload` 和 `/detect` 接口。
- 测试接口：可以用 Postman 发请求试试。
- 检查 `app.log` 如果有问题。

## 注意事项
- 确保 `model/` 文件夹里的 `.pt` 文件存在。
- Mac 上路径要改成斜杠，比如 `/Users/username/scana-main/uploads`。