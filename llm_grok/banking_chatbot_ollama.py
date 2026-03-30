# banking_chatbot_ollama.py
import os
import time
from typing import Dict, List
import ollama

# ────────────────────────────────────────────────
#          BANKING KNOWLEDGE BASE (simple version)
# ────────────────────────────────────────────────

BANK_INFO = {
    "account balance": "You can check your account balance 24/7 through mobile banking, internet banking, or by calling our IVR at *1234#.",
    "transfer money": "To transfer money:\n1. Open the app\n2. Go to 'Fund Transfer'\n3. Choose IMPS/NEFT/RTGS/UPI\n4. Enter beneficiary details or scan QR\n5. Confirm with MPIN/OTP",
    "check statement": "You can download your last 6 months statement from the mobile app → Menu → Statements → Download PDF",
    "debit card lost": "If your debit card is lost/stolen:\n1. Immediately block it via app → Cards → Block Card\n2. Or call 1800-XXX-XXXX (24×7)",
    "credit card payment": "Credit card payment due date is usually 20–25 days after statement generation. You can pay via app, netbanking, UPI, or at branch.",
    "change pin": "You can change ATM PIN via:\n• Our mobile app → Cards → Change PIN\n• ATM machine using old PIN + OTP option",
    "open fixed deposit": "Minimum FD amount is ₹5,000. Tenure 7 days to 10 years. Highest interest for senior citizens.",
    "loan status": "You can check active loan / EMI status in the app → Loans section.",
    "customer care": "Toll-free number: 1800-123-4567\nEmail: support@yourbank.com\nWorking hours: 8 AM – 10 PM",
    "branch locator": "Find nearest branch / ATM → Mobile app → Locate Us or visit www.yourbank.com/locator",
}

GREETING_RESPONSES = [
    "Hello! Welcome to YourBank chatbot 🏦 How can I help you today?",
    "Hi there! I'm your virtual banking assistant. What can I do for you?",
    "Good to see you! How may I assist with your banking needs today?",
]

ERROR_RESPONSES = [
    "I'm sorry, I didn't quite understand that. Could you please rephrase?",
    "Hmm... I'm not sure about that one. Can you ask in another way?",
    "I'm still learning! Could you try asking differently?",
]

# ────────────────────────────────────────────────
#                MAIN CHATBOT LOGIC
# ────────────────────────────────────────────────


def get_intent(user_input: str) -> str:
    """Very simple keyword-based intent detection"""
    text = user_input.lower().strip()

    if any(
        word in text for word in ["balance", "amount", "how much money", "remaining"]
    ):
        return "account balance"

    if any(
        word in text
        for word in ["transfer", "send money", "pay to", "upi", "imps", "neft"]
    ):
        return "transfer money"

    if "statement" in text or "transaction history" in text:
        return "check statement"

    if "lost" in text and ("card" in text or "debit" in text or "atm" in text):
        return "debit card lost"

    if "credit card" in text and ("pay" in text or "payment" in text or "due" in text):
        return "credit card payment"

    if "pin" in text and ("change" in text or "reset" in text):
        return "change pin"

    if "fixed deposit" in text or "fd" in text or "term deposit" in text:
        return "open fixed deposit"

    if "loan" in text and ("status" in text or "emi" in text or "outstanding" in text):
        return "loan status"

    if (
        "customer care" in text
        or "support" in text
        or "helpdesk" in text
        or "contact" in text
    ):
        return "customer care"

    if (
        "branch" in text
        or "atm" in text
        and ("find" in text or "nearest" in text or "locate" in text)
    ):
        return "branch locator"

    return None


def get_answer(intent: str | None, user_input: str) -> str:
    if intent and intent in BANK_INFO:
        return BANK_INFO[intent]

    # Fallback to LLM when we don't have direct match
    try:
        response = ollama.chat(
            model="llama3.2:3b",  # or 'phi4:mini', 'gemma3:4b', 'qwen2.5:3b' etc.
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful, polite and concise banking assistant.
                    Answer only banking related questions.
                    Do NOT give financial advice.
                    Do NOT ask for OTP, PIN, CVV, password or any sensitive information.
                    Keep answers short, clear and safe.""",
                },
                {"role": "user", "content": user_input},
            ],
            options={
                "temperature": 0.1,
                "num_predict": 220,
            },
        )
        return response["message"]["content"].strip()

    except Exception as e:
        print(f"LLM error: {e}")
        return (
            "I'm having trouble connecting to my brain right now 😅 Please try again."
        )


def chatbot():
    print("\n" + "=" * 50)
    print("   Welcome to YourBank AI Assistant  🏦")
    print("=" * 50)
    print("Type 'exit', 'quit' or 'bye' to end the chat\n")

    # Optional: random greeting
    import random

    print(random.choice(GREETING_RESPONSES))
    print()

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("\nThank you for banking with us! Have a great day! 👋")
                break

            if not user_input:
                continue

            print("Thinking...", end="", flush=True)

            intent = get_intent(user_input)
            answer = get_answer(intent, user_input)

            # simulate thinking time
            time.sleep(0.4)
            print("\r" + " " * 12 + "\r", end="")  # clear thinking

            print(f"BankBot: {answer}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("BankBot: Sorry, something went wrong. Please try again.\n")


if __name__ == "__main__":
    print("Starting YourBank Chatbot...\n")
    print("Make sure Ollama is running with a small model (3B–8B recommended)")
    print("Example models: llama3.2:3b, phi4:mini, gemma3:4b, qwen2.5:3b\n")

    chatbot()
