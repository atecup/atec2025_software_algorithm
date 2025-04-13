
import requests
import ollama


def call_api_vlm(prompt, base64_image=None):


    ollama_client = ollama.Client()

    response = ollama_client.chat(model='gemma3:27b',
    messages=[{
        'role': 'system',
        'content': prompt,
        },
        {
            'role': 'user',
            'content':"Robot's observation",
            "images": [base64_image]
        }
    ]
    )

    return response['message']['content']







def call_api_llm(prompt):

    input_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    ollama_client = ollama.Client()
    response = ollama_client.chat(model='gemma3:27b',
    messages=input_messages
    )

    return response['message']['content']





