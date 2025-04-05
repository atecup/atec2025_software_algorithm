from openai import OpenAI
import requests



def call_api_vlm(prompt, base64_image=None):

    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=""
    )

    response = client.chat.completions.create(
        model='qwen2.5-vl-72b-instruct',
        # max_tokens=512,
        messages=[
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }]
            },
        ],

    )

    return response.choices[0].message.content






def call_api_llm(prompt):

    input_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    url = "https://cloud.infini-ai.com/maas/qwen2.5-72b-instruct/nvidia/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f""  # 使用轮流的API密钥
    }

    payload = {
        "model": 'qwen2.5-72b-instruct',
        "messages": input_messages,
        "temperature": 0.6,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()  # 检查请求是否成功
    response_json = response.json()
    response.close()

    return response_json["choices"][0]["message"]["content"]




# def call_api_llm(prompt):

#     input_messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]

#     url = "https://api.siliconflow.cn/v1/chat/completions"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer sk-mdybrmalhqgxobuzouluriezvvgiujmunoyihvhhrfvjtvrv"  # 使用轮流的API密钥
#     }

#     payload = {
#         "model": 'Qwen/Qwen2.5-72B-Instruct',
#         "messages": input_messages,
#         "temperature": 0.6,
#     }

#     response = requests.post(url, headers=headers, json=payload, timeout=60)
#     response.raise_for_status()  # 检查请求是否成功
#     response_json = response.json()
#     response.close()

#     return response_json["choices"][0]["message"]["content"]

