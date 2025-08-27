# API使用文档

---

## **1. 用户接口**

### **1.1 用户注册**

* **URL**: `/api/register`
* **方法**: `POST`
* **请求体**:

```json
{
  "username": "test_user",
  "password": "123456"
}
```

* **响应示例**:

```json
{
  "message": "Registration successful"
}
```

* **状态码**:

  * `201` 注册成功
  * `400` 用户名已存在或缺少参数

---

### **1.2 用户登录**

* **URL**: `/api/login`
* **方法**: `POST`
* **请求体**:

```json
{
  "username": "test_user",
  "password": "123456"
}
```

* **响应示例**:

```json
{
  "message": "Login successful",
  "username": "test_user"
}
```

* **状态码**:

  * `200` 登录成功
  * `400` 参数缺失
  * `401` 用户名或密码错误

---

### **1.3 用户登出**

* **URL**: `/api/logout`
* **方法**: `POST`
* **响应示例**:

```json
{
  "message": "Logged out successfully"
}
```

---

## **2. 扫描接口**

### **2.1 启动扫描任务**

* **URL**: `/api/scan/start`
* **方法**: `POST`
* **认证**: 需已登录用户（Session）
* **支持两种提交方式**:

#### (A) 上传文件

* **表单**: `multipart/form-data`
* **字段**:

  * `file`: 上传的源代码文件

#### (B) 直接提交代码

* **请求体**:

```json
{
  "filename": "test.py",
  "code": "print('hello world')"
}
```

* **响应示例**:

```json
{
  "message": "Scan started",
  "task_id": "e6a3d9d8-3fa6-4c8a-8b9a-8f2bde3a1a1c"
}
```

* **状态码**:

  * `202` 扫描已启动
  * `400` 缺少参数或文件名为空
  * `401` 未登录

---

### **2.2 查询任务状态**

* **URL**: `/api/scan/status/<task_id>`
* **方法**: `GET`
* **响应示例**:

```json
{
  "task_id": "e6a3d9d8-3fa6-4c8a-8b9a-8f2bde3a1a1c",
  "filename": "test.py",
  "status": "done",
  "progress": 1.0,
  "lines_scanned": 5,
  "estimated_time_left": 0,
  "vul_distribution": {"predicted_vulnerability": 1},
  "vul_list": [
    {
      "id": "9a1f8c8e-91b2-4e77-8b8e-1e63a41db8b6",
      "line": 1,
      "type": "predicted_vulnerability",
      "severity": "medium",
      "description": "Predicted vulnerability with probability 0.56"
    }
  ],
  "vul_prob": 0.56,
  "embedding_shape": [1, 100, 300],
  "start_time": "2025-08-27T12:00:00",
  "end_time": "2025-08-27T12:00:05",
  "duration": "5.0s",
  "error_msg": null
}
```

* **状态码**:

  * `200` 查询成功
  * `401` 未登录
  * `403` 无权访问
  * `404` 任务不存在

---

### **2.3 查询历史记录**

* **URL**: `/api/scan/history`
* **方法**: `GET`
* **响应示例**:

```json
[
  {
    "task_id": "...",
    "filename": "...",
    "status": "done",
    "progress": 1.0,
    "vul_distribution": {...},
    "vul_list": [...],
    "start_time": "...",
    "end_time": "...",
    "duration": "..."
  },
  ...
]
```

* **状态码**:

  * `200` 查询成功
  * `401` 未登录

---

## **3. 状态说明**

* **status 字段**:

  * `pending`：任务已提交，等待处理
  * `running`：任务正在扫描
  * `done`：扫描完成
  * `failed`：任务失败，`error_msg` 包含错误信息

---

## **4. 返回格式**

所有接口返回均为 JSON 格式，包含结果或错误信息。

---
