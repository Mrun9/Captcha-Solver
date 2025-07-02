import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print("Using API Key:", api_key)


url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama3-70b-8192",  # or use "mixtral-8x7b-32768", "gemma-7b-it", etc.
    "messages": [
        {"role": "user", "content": "Tell me a joke about AI."}
    ],
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    content = response.json()
    print(content["choices"][0]["message"]["content"])
else:
    print("Error:", response.status_code, response.text)
