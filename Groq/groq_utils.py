# groq_utils.py

import os
import base64
import requests
from PIL import Image

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime_type(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    return {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
    }.get(ext, 'image/png')

def ask_llama_unscrambled(image_paths):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Few-shot examples
    few_shot = []
    few_shot_examples = [
        ("scrambled", "examples/scrambled1.png"),
        ("unscrambled", "examples/unscrambled1.png"),
    ]
    for label, path in few_shot_examples:
        base64_img = encode_image_to_base64(path)
        few_shot.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{get_mime_type(path)};base64,{base64_img}"}},
                {"type": "text", "text": "Is this scrambled or unscrambled?"}
            ]
        })
        few_shot.append({"role": "assistant", "content": label})

    # Ask for the correct image
    predictions = []
    for i, path in enumerate(image_paths):
        base64_img = encode_image_to_base64(path)

        messages = few_shot + [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{get_mime_type(path)};base64,{base64_img}"}},
                    {"type": "text", "text": "Is this image scrambled or unscrambled? Respond with ONLY 'scrambled' or 'unscrambled'."}
                ]
            }
        ]

        payload = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 10
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            text = response.json()['choices'][0]['message']['content'].strip().lower()
            predictions.append(text)
        else:
            print(f"Error on image {i}: {response.text}")
            predictions.append("error")

    # Find first 'unscrambled' index
    for i, pred in enumerate(predictions):
        if pred == "unscrambled":
            return i

    return 5  # fallback
