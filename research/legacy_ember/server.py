import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import requests
import json
import traceback

# ================ 1. 创建应用 ================
app = FastAPI(title="内容风险检测API（最终稳定版）")

# ================ 2. 请求格式定义 ================
class RequestData(BaseModel):
    task_type: str
    params: Dict[str, Any]
    extra: Optional[str] = None

# ================ 3. 风险检测（超时=有害） ================
def check_content_risk(content: str):
    try:
        print(f"[DEBUG] 检测内容: {content}")

        API_KEY = os.environ.get("API_KEY", "")
        MODEL = "doubao-seed-1-6-251015"
        URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": """你是专业内容安全审核专家。
请检测文本是否违规：涉政、民族、宗教、暴力、色情、歧视、违法等。
必须返回标准JSON，包含4个字段：
risk_level: 无风险/低风险/中风险/高风险
risk_type: list类型，风险类型，无则空列表
confidence: 0~1 浮点数
suggestion: 字符串建议"""
                },
                {"role": "user", "content": content}
            ],
            "temperature": 0.1,
            "max_tokens": 512
        }

        # 调用
        resp = requests.post(URL, headers=headers, json=payload, timeout=15)
        result = resp.json()
        reply = result["choices"][0]["message"]["content"].strip()
        return json.loads(reply)

    # 超时 → 直接判定有害
    except requests.exceptions.Timeout:
        print("[ERROR] API超时 → 判定为有害内容")
        return {
            "risk_level": "高风险",
            "risk_type": ["有害内容（API超时判定）"],
            "confidence": 0.99,
            "suggestion": "内容存在安全风险，禁止使用"
        }

    # 其他任何错误 → 都判定有害
    except Exception as e:
        print(f"[ERROR] 检测失败: {e} → 判定为有害")
        return {
            "risk_level": "高风险",
            "risk_type": ["有害内容（API异常判定）"],
            "confidence": 0.99,
            "suggestion": "内容存在安全风险，禁止使用"
        }

# ================ 4. 接口（永远返回200，绝不报错） ================
@app.post("/api/handle")
def handle_api(data: RequestData):
    # 永远不抛错！永远返回200
    try:
        content = data.params.get("content", "")
        risk_result = check_content_risk(content)

        return {
            "code": 200,
            "msg": "检测完成",
            "data": {
                "task": data.task_type,
                "your_content": content,
                "risk_check": risk_result
            }
        }

    # 终极兜底：任何意外 → 依然返回200+有害
    except:
        return {
            "code": 200,
            "msg": "检测完成（服务异常）",
            "data": {
                "task": data.task_type,
                "your_content": data.params.get("content", ""),
                "risk_check": {
                    "risk_level": "高风险",
                    "risk_type": ["服务异常，自动判定有害"],
                    "confidence": 0.99,
                    "suggestion": "内容存在风险"
                }
            }
        }

# ================ 5. 启动 ================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=80)