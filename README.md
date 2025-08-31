## **SCANA API 使用文档**

### **服务概述**

SCANA 是一个智能合约漏洞检测平台，提供注册、登录、代码提交、扫描状态查询及历史记录功能。
后端使用预训练的 **BLSTM + Attention** 模型预测潜在漏洞，并返回扫描结果。

---

## **基础信息**

* **Base URL**: `http://<server>:8000/api`
* **认证方式**: 会话（Session Cookie），需要先登录后再调用扫描相关接口。
* **数据格式**: 所有请求与响应均使用 `application/json`，文件上传使用 `multipart/form-data`。

---

## **接口一览**

### **1. 用户注册**

* **URL**: `/register`
* **方法**: `POST`
* **请求参数（JSON）**:

  ```json
  {
    "username": "user123",
    "password": "securePassword"
  }
  ```
* **响应**:

  * **201**: `{"message": "Registration successful."}`
  * **409**: `{"error": "Username 'user123' already exists."}`

---

### **2. 用户登录**

* **URL**: `/login`
* **方法**: `POST`
* **请求参数（JSON）**:

  ```json
  {
    "username": "user123",
    "password": "securePassword"
  }
  ```
* **响应**:

  * **200**:

    ```json
    {
      "message": "Login successful.",
      "username": "user123"
    }
    ```
  * **401**: `{"error": "Invalid username or password."}`

---

### **3. 用户登出**

* **URL**: `/logout`
* **方法**: `POST`
* **响应**:

  * **200**: `{"message": "Logged out successfully."}`

---

### **4. 启动代码扫描**

* **URL**: `/scan/start`
* **方法**: `POST`
* **请求类型**: `multipart/form-data` 或 `application/json`
* **请求参数**:

  * **方式一：文件上传**

    * 参数：`file`（必填）
  * **方式二：直接传代码**

    ```json
    {
      "filename": "contract.sol",
      "code": "pragma solidity ^0.8.0; ..."
    }
    ```
* **响应**:

  * **202**:

    ```json
    {
      "message": "Scan task initiated successfully.",
      "task_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```
  * **401**: `{"error": "Authentication required."}`

---

### **5. 查询扫描状态**

* **URL**: `/scan/status/<task_id>`
* **方法**: `GET`
* **响应示例**:

  ```json
  {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "contract.sol",
    "status": "done",
    "progress": 1.0,
    "lines_scanned": 120,
    "vul_distribution": {
      "high": 1,
      "medium": 0,
      "low": 0
    },
    "vul_list": [
      {
        "id": "aabbccdd-1122-3344-5566-77889900aabb",
        "line": 1,
        "type": "predicted_reentrancy",
        "severity": "high",
        "description": "A potential vulnerability was detected with a confidence of 92.50%.",
        "confidence": "0.9250"
      }
    ],
    "vul_prob": 0.925,
    "embedding_shape": [1, 300, 1700],
    "start_time": "2025-08-27T14:32:01.123456",
    "end_time": "2025-08-27T14:32:05.987654",
    "duration": "4.86s",
    "error_msg": null
  }
  ```
* **错误码**:

  * **401**: 未登录
  * **403**: 访问非本人任务
  * **404**: 任务不存在

---

### **6. 查询扫描历史**

* **URL**: `/scan/history`
* **方法**: `GET`
* **响应**:

  ```json
  [
    {
      "task_id": "...",
      "filename": "contract1.sol",
      "status": "done",
      "progress": 1.0,
      "lines_scanned": 80,
      ...
    },
    {
      "task_id": "...",
      "filename": "contract2.sol",
      "status": "failed",
      ...
    }
  ]
  ```

---

## **状态说明**

* **pending**: 任务已创建，等待执行
* **running**: 扫描中
* **done**: 扫描完成，结果可查询
* **failed**: 扫描失败，`error_msg` 提供错误信息
