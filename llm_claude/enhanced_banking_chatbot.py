"""
Enhanced Banking Chatbot with Rule-Based Fallback
This version includes pattern matching for instant responses to common queries
"""

import json
import re
from typing import Optional
import requests

class EnhancedBankingChatbot:
    def __init__(self, model_provider="ollama", use_fallback=True):
        """Initialize the enhanced banking chatbot"""
        self.model_provider = model_provider
        self.use_fallback = use_fallback
        self.conversation_history = []
        
        # Banking data
        self.banking_info = {
            "savings_rate": "4% per annum",
            "fd_rates": "6-7% interest based on tenure",
            "personal_loan_rate": "9-12% per annum",
            "home_loan_rate": "8-10% per annum",
            "operating_hours": "Monday-Friday: 9 AM - 5 PM, Saturday: 9 AM - 1 PM",
            "customer_service": "1-800-BANK-HELP"
        }
        
        # Rule-based patterns for instant responses
        self.patterns = [
            {
                "keywords": ["balance", "minimum balance"],
                "response": "Our minimum balance requirements are:\n- Savings Account: $500\n- Current Account: No minimum balance\n\nWould you like to know more about our account types?"
            },
            {
                "keywords": ["savings", "savings account"],
                "response": f"Our Savings Account offers {self.banking_info['savings_rate']} interest. Features include:\n- Competitive rates\n- Free ATM withdrawals\n- Online banking\n\nInterested in opening an account?"
            },
            {
                "keywords": ["hours", "operating hours", "timing", "when open"],
                "response": f"Banking hours: {self.banking_info['operating_hours']}\n\nFor 24/7 help, call {self.banking_info['customer_service']}"
            },
            {
                "keywords": ["contact", "customer service", "support", "phone"],
                "response": f"Contact us at: {self.banking_info['customer_service']}\n\nAvailable during: {self.banking_info['operating_hours']}"
            },
            {
                "keywords": ["loan", "personal loan", "borrow"],
                "response": f"Personal Loans available:\n- Rate: {self.banking_info['personal_loan_rate']}\n- Amount: $5,000 - $100,000\n- Tenure: 1-5 years\n\nWant to apply?"
            },
            {
                "keywords": ["home loan", "housing loan", "mortgage"],
                "response": f"Home Loan features:\n- Rate: {self.banking_info['home_loan_rate']}\n- Up to $500,000\n- 30-year tenure\n\nCall {self.banking_info['customer_service']} to discuss."
            },
            {
                "keywords": ["credit card", "card"],
                "response": "Credit Cards offer:\n- Annual fee from $50\n- Rewards program\n- 45-day interest-free period\n\nApply today!"
            },
            {
                "keywords": ["fixed deposit", "fd"],
                "response": f"Fixed Deposits:\n- Rate: {self.banking_info['fd_rates']}\n- Min deposit: $1,000\n- Flexible tenure\n\nShall I help you open one?"
            },
            {
                "keywords": ["transfer", "send money"],
                "response": "Transfer funds via:\n- Online Banking (24/7)\n- Mobile App\n- Branch visit\n- ATM\n\nYou'll need account number and IFSC code."
            }
        ]
        
        self.banking_context = f"""You are a banking assistant. 

Services:
- Savings: {self.banking_info['savings_rate']}
- Personal Loan: {self.banking_info['personal_loan_rate']}
- Home Loan: {self.banking_info['home_loan_rate']}
- Fixed Deposit: {self.banking_info['fd_rates']}

Hours: {self.banking_info['operating_hours']}
Support: {self.banking_info['customer_service']}

Be helpful and concise."""
        
        self.hf_api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        self.hf_headers = {"Content-Type": "application/json"}
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def check_patterns(self, question: str) -> Optional[str]:
        """Check if question matches predefined patterns"""
        if not self.use_fallback:
            return None
        
        question_lower = question.lower()
        for pattern in self.patterns:
            if any(kw in question_lower for kw in pattern["keywords"]):
                return pattern["response"]
        return None
    
    def query_ollama(self, prompt: str) -> str:
        """Query Ollama local model"""
        try:
            payload = {
                "model": "llama2",
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["response"]
            return f"Error: Status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return "Error: Ollama not running. Start with 'ollama serve'"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def query_huggingface(self, prompt: str) -> str:
        """Query Hugging Face API"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            response = requests.post(self.hf_api_url, headers=self.hf_headers, 
                                    json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "No response")
                return str(result)
            elif response.status_code == 503:
                return "Model loading. Please retry in a moment."
            return f"Error: Status {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_prompt(self, user_question: str) -> str:
        """Create prompt for LLM"""
        history = ""
        if self.conversation_history:
            history = "Recent chat:\n"
            for ex in self.conversation_history[-2:]:
                history += f"User: {ex['user']}\nBot: {ex['assistant']}\n"
        
        return f"""{self.banking_context}

{history}
User: {user_question}

Assistant (be helpful and brief):"""
    
    def get_response(self, user_question: str) -> str:
        """Get chatbot response"""
        # Try pattern matching first
        pattern_response = self.check_patterns(user_question)
        if pattern_response:
            response = f"[Quick Answer]\n{pattern_response}"
            self.conversation_history.append({
                "user": user_question,
                "assistant": response
            })
            return response
        
        # Use LLM
        prompt = self.create_prompt(user_question)
        
        if self.model_provider == "ollama":
            response = self.query_ollama(prompt)
        elif self.model_provider == "huggingface":
            response = self.query_huggingface(prompt)
        else:
            response = "Invalid provider"
        
        self.conversation_history.append({
            "user": user_question,
            "assistant": response
        })
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("History cleared.")


def main():
    """Run the chatbot"""
    print("=" * 60)
    print("Welcome to Enhanced Banking Chatbot!")
    print("=" * 60)
    print("\nFeatures:")
    print("✓ Instant answers for common questions")
    print("✓ AI-powered responses for complex queries")
    print("\nChoose LLM provider:")
    print("1. Ollama (Local)")
    print("2. Hugging Face (Online)")
    
    choice = input("\nChoice (1/2): ").strip()
    provider = "ollama" if choice == "1" else "huggingface"
    
    if provider == "ollama":
        print("\n📌 Using Ollama")
        print("Ensure: 'ollama serve' is running")
        print("Model installed: 'ollama pull llama2'")
    else:
        print("\n📌 Using Hugging Face API")
        print("First request may be slow (model loading)")
    
    chatbot = EnhancedBankingChatbot(model_provider=provider)
    
    print("\n" + "=" * 60)
    print("Ready! Ask your banking questions.")
    print("Commands: 'quit' to exit, 'clear' to reset")
    print("=" * 60 + "\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you! Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            chatbot.clear_history()
            continue
        
        print("\nBot: ", end="", flush=True)
        response = chatbot.get_response(user_input)
        print(response + "\n")


if __name__ == "__main__":
    main()
