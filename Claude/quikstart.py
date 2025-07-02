import anthropic
from config import get_api_key, get_model_name
client = anthropic.Anthropic(api_key = get_api_key())

message = client.messages.create(
    model=get_model_name(),
    max_tokens=1000,
    temperature=1,
    system="You are a world-class poet. Respond only with short poems.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Why is the ocean salty?"
                }
            ]
        }
    ]
)
print(message.content)