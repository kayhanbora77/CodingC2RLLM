"""
Banking Chatbot using Free LLM Models
Supports both Ollama (local) and Hugging Face API (free tier)
"""

import json
from typing import List, Dict
import requests

class BankingChatbot:
    def __init__(self, model_provider="ollama"):
        """
        Initialize the banking chatbot
        
        Args:
            model_provider: "ollama" for local models or "huggingface" for HF API
        """
        self.model_provider = model_provider
        self.conversation_history = []
        
        # Banking knowledge base
        self.banking_context = """
        You are a helpful banking assistant. You can help with:
        - Account information and balance inquiries
        - Transaction history
        - Fund transfers
        - Loan information
        - Credit card services
        - Fixed deposits and savings
        - Customer support
        
        Banking Services Available:
        - Savings Account: 4% interest per annum
        - Current Account: No interest, no minimum balance
        - Fixed Deposit: 6-7% interest based on tenure
        - Personal Loan: 9-12% interest
        - Home Loan: 8-10% interest
        - Credit Cards: Annual fee starting from $50
        
        Operating Hours: Monday-Friday 9 AM - 5 PM, Saturday 9 AM - 1 PM
        Customer Service: 1-800-BANK-HELP
        """
        
        # HuggingFace API setup (if using HF)
        self.hf_api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        self.hf_headers = {
            "Content-Type": "application/json"
        }
        
        # Ollama API setup (if using Ollama)
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def query_ollama(self, prompt: str) -> str:
        """Query Ollama local model"""
        try:
            payload = {
                "model": "llama2",  # or "mistral", "phi", etc.
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: Unable to connect to Ollama (Status {response.status_code})"
        
        except requests.exceptions.ConnectionError:
            return "Error: Ollama is not running. Please start Ollama first with 'ollama serve'"
        except Exception as e:
            return f"Error querying Ollama: {str(e)}"
    
    def query_huggingface(self, prompt: str) -> str:
        """Query Hugging Face API (free tier)"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                self.hf_api_url,
                headers=self.hf_headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "No response generated")
                return str(result)
            elif response.status_code == 503:
                return "Model is loading on Hugging Face servers. Please wait a moment and try again."
            else:
                return f"Error: API returned status {response.status_code}"
        
        except Exception as e:
            return f"Error querying Hugging Face: {str(e)}"
    
    def create_prompt(self, user_question: str) -> str:
        """Create a structured prompt for the LLM"""
        
        # Add conversation history context
        history_context = ""
        if self.conversation_history:
            history_context = "Previous conversation:\n"
            for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
                history_context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"
        
        prompt = f"""
{self.banking_context}

{history_context}

User Question: {user_question}

Please provide a helpful, accurate, and professional response as a banking assistant. 
Be concise and friendly. If you don't have specific information, guide the user to contact customer service.

Assistant Response:"""
        
        return prompt
    
    def get_response(self, user_question: str) -> str:
        """Get response from the selected LLM provider"""
        
        # Create the prompt
        prompt = self.create_prompt(user_question)
        
        # Query the appropriate model
        if self.model_provider == "ollama":
            response = self.query_ollama(prompt)
        elif self.model_provider == "huggingface":
            response = self.query_huggingface(prompt)
        else:
            response = "Invalid model provider specified"
        
        # Store in conversation history
        self.conversation_history.append({
            "user": user_question,
            "assistant": response
        })
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.")


def main():
    """Main function to run the chatbot"""
    
    print("=" * 60)
    print("Welcome to Banking Chatbot!")
    print("=" * 60)
    print("\nChoose your LLM provider:")
    print("1. Ollama (Local - requires Ollama installed)")
    print("2. Hugging Face (Online - free API, may have rate limits)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        provider = "ollama"
        print("\nUsing Ollama (Local Model)")
        print("Make sure Ollama is running: 'ollama serve'")
        print("And you have a model installed: 'ollama pull llama2'")
    elif choice == "2":
        provider = "huggingface"
        print("\nUsing Hugging Face API (Mistral-7B)")
        print("Note: First request may take time as model loads")
    else:
        print("Invalid choice. Defaulting to Ollama.")
        provider = "ollama"
    
    # Initialize chatbot
    chatbot = BankingChatbot(model_provider=provider)
    
    print("\n" + "=" * 60)
    print("Chatbot is ready! Type your questions below.")
    print("Commands: 'quit' to exit, 'clear' to clear history")
    print("=" * 60 + "\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Check for commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for using Banking Chatbot! Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            chatbot.clear_history()
            continue
        
        # Get response from chatbot
        print("\nBanking Assistant: ", end="", flush=True)
        response = chatbot.get_response(user_input)
        print(response)
        print()


if __name__ == "__main__":
    main()
