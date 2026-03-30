# Banking Chatbot with Free LLM

A Python-based banking chatbot that uses free LLM models to answer customer questions about banking services.

## Features

- ✅ Two LLM options: **Ollama (local)** or **Hugging Face API (online)**
- ✅ Banking knowledge base included
- ✅ Conversation history tracking
- ✅ Professional banking assistant responses
- ✅ Easy to use command-line interface

## Supported Queries

The chatbot can help with:
- Account balance inquiries
- Transaction information
- Fund transfers
- Loan details
- Credit card services
- Fixed deposits and savings
- General banking questions

## Installation

### Prerequisites

**Python 3.8+** is required.

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Choose Your LLM Provider

#### Option A: Ollama (Recommended - Local & Free)

**Advantages:**
- Runs completely offline
- No API limits
- Fast responses
- Privacy-focused

**Installation:**

1. Install Ollama from https://ollama.ai

2. Pull a model (choose one):
```bash
# Lightweight and fast
ollama pull llama2

# Or use Mistral (better quality)
ollama pull mistral

# Or use Phi (smallest, fastest)
ollama pull phi
```

3. Start Ollama server:
```bash
ollama serve
```

4. Keep the server running in a separate terminal

#### Option B: Hugging Face API (Online & Free)

**Advantages:**
- No local installation needed
- Works immediately
- Uses Mistral-7B model

**Note:** The free tier has rate limits and the model may take time to load on first request.

No additional setup required - just run the chatbot!

## Usage

### Run the Chatbot

```bash
python banking_chatbot.py
```

### Choose Your Provider

When prompted, select:
- `1` for Ollama (local)
- `2` for Hugging Face (online)

### Example Conversation

```
You: What are your savings account interest rates?

Banking Assistant: Our savings account offers a competitive interest rate 
of 4% per annum. This is calculated on your daily balance and credited 
to your account quarterly. Would you like to know more about opening a 
savings account?

You: What are your operating hours?

Banking Assistant: Our banking hours are:
- Monday to Friday: 9 AM - 5 PM
- Saturday: 9 AM - 1 PM
- Sunday: Closed

For 24/7 assistance, you can reach our customer service at 1-800-BANK-HELP.

You: quit
```

### Commands

- `quit` or `exit` - Exit the chatbot
- `clear` - Clear conversation history

## Customization

### Update Banking Information

Edit the `banking_context` in `banking_chatbot.py` to update:
- Interest rates
- Service offerings
- Operating hours
- Contact information

### Change LLM Model (Ollama)

In the `query_ollama` method, change the model name:

```python
payload = {
    "model": "mistral",  # Change to "llama2", "phi", etc.
    "prompt": prompt,
    "stream": False
}
```

Available Ollama models:
- `llama2` - 7B parameters, general purpose
- `mistral` - 7B parameters, high quality
- `phi` - 2.7B parameters, lightweight
- `codellama` - 7B parameters, code-focused
- `neural-chat` - 7B parameters, conversational

Install any model with: `ollama pull <model-name>`

### Change HuggingFace Model

Update the `hf_api_url` to use different models:

```python
# Mistral 7B (default)
self.hf_api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# Or try Falcon 7B
self.hf_api_url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
```

## Troubleshooting

### Ollama Issues

**Error: "Ollama is not running"**
- Make sure Ollama server is running: `ollama serve`
- Check if port 11434 is available

**Error: "model not found"**
- Pull the model first: `ollama pull llama2`
- List available models: `ollama list`

### Hugging Face Issues

**Error: "Model is loading"**
- Wait 20-30 seconds for the model to load
- Try again after a moment

**Error: 503 Service Unavailable**
- The model is loading on HF servers
- Wait and retry in a few seconds

**Rate Limiting**
- Free tier has rate limits
- Consider using Ollama for unlimited queries

## Advanced Features

### Add API Key for Hugging Face (Optional)

For better rate limits, get a free API key from https://huggingface.co/settings/tokens

Add to the code:

```python
self.hf_headers = {
    "Authorization": f"Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
```

### Integrate with Database

Connect to a real banking database by modifying the chatbot to query actual account information:

```python
def get_account_balance(self, account_number):
    # Connect to your database
    # Fetch real balance
    return balance
```

## Project Structure

```
banking_chatbot/
├── banking_chatbot.py    # Main chatbot code
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Security Notes

⚠️ **Important:** This is a demo chatbot. For production use:
- Implement proper authentication
- Use secure database connections
- Add input validation
- Encrypt sensitive data
- Add logging and monitoring
- Implement rate limiting

## License

Free to use and modify for educational and commercial purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review Ollama docs: https://ollama.ai
3. Review Hugging Face docs: https://huggingface.co/docs

## Contributing

Feel free to enhance this chatbot by:
- Adding more banking services
- Improving the knowledge base
- Adding voice support
- Creating a web interface
- Integrating with real banking APIs
