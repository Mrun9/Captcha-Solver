# groq_utils.py

import base64
import requests
from config import get_api_key, get_model_name
from PIL import Image
import io

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')



def compress_and_resize_image(image_path, max_size=(300, 100)):
    """
    Resize and compress the image to reduce payload size.
    Returns: base64-encoded image string
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail(max_size)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=60)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    
def ask_llama_unscrambled(images):
    api_key = get_api_key()
    model = get_model_name()
    
    prompt_text = """You are given a group of images from a captcha row. Only one is unscrambled (visually coherent). The rest are scrambled (shuffled parts). Identify which image is unscrambled. Respond with just the index (0 to 5). Nothing else."""

    # Create image content blocks
    image_blocks = []
    for i, img_path in enumerate(images):
        base64_data = compress_and_resize_image(img_path)
        image_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_data}"
            }
        })

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Combine all messages
    messages = [
        {
            "role": "user",
            "content": image_blocks + [
                {
                    "type": "text",
                    "text": prompt_text
                }
            ]
        }
    ]

    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0.2
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()

    text = response.json()['choices'][0]['message']['content'].strip()

    # Extract index from response
    for token in text.split():
        if token.isdigit() and int(token) in range(6):
            return int(token)

    return None
