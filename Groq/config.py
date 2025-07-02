# config.py
import os

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def get_api_key():
    return api_key  # Replace with your actual API key

def get_model_name():
    return "llama3-8b-8192"#"meta-llama/llama-4-scout-17b-16e-instruct"  # or meta-llama/llama-4-maverick-17b-128e-instruct
