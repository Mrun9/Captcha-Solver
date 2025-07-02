import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def chat_function(prompt, model):
    openai.api_key = os.getenv('OPENAI_API_KEY')

    response = openai.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

if __name__ == "__main__":
    model = 'gpt-3.5-turbo'  # Default model, can be changed as needed
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            print("Exiting chat.")
            break
        chat_function(prompt, model)
        print()  # Print a newline for better readability