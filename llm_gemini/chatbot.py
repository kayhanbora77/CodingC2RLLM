import ollama

# 1. Simple Knowledge Base (In a real app, this would be a PDF or Database)
bank_data = {
    "opening_hours": "Our bank is open Monday to Friday, 9:00 AM to 5:00 PM.",
    "balance_check": "To check your balance, log into the mobile app or visit an ATM.",
    "lost_card": "If you lost your card, call 1-800-BANK-HELP immediately to block it.",
    "interest_rates": "Our current savings interest rate is 2.5% per annum.",
}


def get_bank_context(user_query):
    # Basic keyword search (simplified RAG)
    for key in bank_data:
        if key in user_query.lower():
            return bank_data[key]
    return "Standard banking procedures apply."


def chatbot_response(user_input):
    context = get_bank_context(user_input)

    # Constructing the prompt for the LLM
    prompt = f"""
    You are a helpful banking assistant. 
    Use the following information to answer the customer's question.
    Information: {context}
    Customer: {user_input}
    Assistant:"""

    response = ollama.generate(model="llama3.2:1b", prompt=prompt)
    return response["response"]


# --- Test the Chatbot ---
print("BankBot: Hello! How can I help you today? (type 'exit' to quit)")
while True:
    user_q = input("You: ")
    if user_q.lower() == "exit":
        break
    print("BankBot:", chatbot_response(user_q))
