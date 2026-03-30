import sys
from llama_cpp import Llama

# CONFIGURATION
# n_gpu_layers: -1 means use CPU (Safe for Lenovo G580)
# n_ctx: Context window (how much text it remembers). 2048 is good for small models.
# n_threads: Set to 2 or 4 to prevent your laptop from freezing.
llm = Llama(
    model_path="./model.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    n_threads=2,  # Adjust based on your CPU (usually 2 or 4 is safe)
    verbose=False,
)


def load_knowledge():
    """Reads the text file to 'train' the bot with context."""
    try:
        with open("bank_knowledge.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        print("Error: bank_knowledge.txt not found!")
        return ""


def generate_response(prompt, context):
    """Generates a response using the tiny LLM."""

    # SYSTEM PROMPT: This tells the AI who it is
    system_message = (
        "You are a helpful, polite banking assistant for SecureBank. "
        "Use the provided CONTEXT to answer the user's question. "
        "If the answer is not in the context, say you don't know."
    )

    # Combine everything into the prompt format the model expects
    full_prompt = f"""
    SYSTEM: {system_message}
    
    CONTEXT:
    {context}
    
    USER: {prompt}
    ASSISTANT:
    """

    # Generate response
    output = llm(
        full_prompt,  # Prompt
        max_tokens=150,  # Limit answer length
        stop=["USER:", "SYSTEM:"],  # Stop generating if it mimics the user
        echo=False,
        temperature=0.7,  # Creativity (0.1 = strict, 1.0 = creative)
    )

    return output["choices"][0]["text"].strip()


def main():
    print("--- SecureBank AI Chatbot (Lenovo G580 Edition) ---")
    print("Loading knowledge base...")
    knowledge_base = load_knowledge()
    print("Bot is ready! (Type 'quit' to exit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        print("Bot is thinking... (this may take a few seconds on weak PCs)")

        try:
            response = generate_response(user_input, knowledge_base)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
