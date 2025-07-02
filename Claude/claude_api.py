import anthropic

def ask_claude(question, api_key, model):
    """Send question to Claude and return response"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": question}]
        )
        
        return message.content[0].text
        
    except Exception as e:
        return f"Error: {e}"