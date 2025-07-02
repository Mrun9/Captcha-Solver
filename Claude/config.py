import os
from dotenv import load_dotenv

def get_api_key():
    """Get the API key from environment variable"""
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        api_key = input("Enter your Anthropic API key: ")
    return api_key

def get_model_name():
    """Get the model name from environment variable or default"""
    load_dotenv()   # Load environment variables from .env file
    model_name = os.getenv('MODEL_NAME')
    return model_name