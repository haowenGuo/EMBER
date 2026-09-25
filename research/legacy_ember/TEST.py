import requests
import config

GPT_API_CONFIG = {
    "url": "https://api.gptsapi.net/v1/messages",
    "headers": {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.OPENAI_API_KEY}"
    }
}
payload = {
    "model": config.openai_model_name,
    "messages": [
        {
            "role": "user",
            "content": "你好，请介绍一下自己"
        }
    ],
    "max_tokens": 4000
}

try:
    response = requests.post(
        url=GPT_API_CONFIG["url"],
        headers=GPT_API_CONFIG["headers"],
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    res = response.json()
    print("调用成功！回复：", res["content"][0]["text"])
except Exception as e:
    print(f"调用GPT API失败: {e}")
    # 核心新增：打印接口返回的原始错误信息（这是定位问题的关键）
    if 'response' in locals():
        print(f"接口返回的详细错误：{response.text}")
        print(f"响应状态码：{response.status_code}")