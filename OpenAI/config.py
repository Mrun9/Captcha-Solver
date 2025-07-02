import os
from dotenv import load_dotenv

def get_api_key():
    """Get the API key from environment variable"""
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv('OPENAI_SECRET_KEY')
    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
    return api_key