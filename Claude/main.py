from config import get_api_key, get_model_name
from claude_api import ask_claude

def init():
    """Initialize the app"""
    print("Starting Claude chatbot...")
    api_key = get_api_key()
    model = get_model_name()
    return api_key, model

def main():
    # Initialize
    api_key, model = init()
    
    print("Ask me anything! (type 'quit' to exit)")
    
    while True:
        question = input("\nYou: ")
        
        if question.lower() == 'quit':
            break
            
        # Call Claude API
        answer = ask_claude(question, api_key, model)
        print(f"Claude: {answer}")

if __name__ == "__main__":
    main()