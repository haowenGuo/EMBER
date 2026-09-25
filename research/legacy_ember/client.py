import requests
import json

# ========== 配置区（你只需要改这里） ==========
# 本地测试地址
API_URL = "http://127.0.0.1:80/api/handle"
#API_URL ="http://53db5271.r30.cpolar.top/api/handl"
# 公网地址（cpolar 给你的地址）
# API_URL = "http://4c1ac08.r30.cpolar.top/api/handle"

# =============================================

def call_api(task_type, params, extra=None):
    """
    调用API接口
    :param task_type: 任务类型
    :param params: 参数字典，必须包含 'content' 字段
    :param extra: 可选额外信息
    :return: API返回结果
    """
    # 构造要发送的 JSON 数据
    payload = {
        "task_type": task_type,
        "params": params,
        "extra": extra
    }

    try:
        # 发送 POST 请求（JSON 格式）
        response = requests.post(
            url=API_URL,
            json=payload,
            timeout=10
        )

        # 判断请求是否成功
        if response.status_code == 200:
            result = response.json()
            print("✅ API调用成功！")
            print("📄 返回结果：")
            print(json.dumps(result, indent=4, ensure_ascii=False))
            return result
        else:
            print(f"❌ 请求失败，状态码：{response.status_code}")
            print("错误信息：", response.text)
            return None

    except Exception as e:
        print(f"❌ 网络/请求异常：{str(e)}")
        return None


# ========== 测试调用 ==========
if __name__ == "__main__":
    # 演示1：包含风险内容的示例
    risky_params = {
        "content": "高山族就是被赶到山上的一个民族。"
    }

    print("=== 演示1：检测风险内容 ===")
    call_api(
        task_type="risk_check",
        params=risky_params,
        extra="演示风险内容检测"
    )

    print("\n" + "="*50 + "\n")

    # 演示2：正常内容的示例
    safe_params = {
        "content": "今天天气很好，适合出去散步和运动。"
    }

    print("=== 演示2：检测正常内容 ===")
    call_api(
        task_type="risk_check",
        params=safe_params,
        extra="演示正常内容检测"
    )