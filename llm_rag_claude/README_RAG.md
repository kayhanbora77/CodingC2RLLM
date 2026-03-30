# Banking Chatbot with RAG (Retrieval-Augmented Generation)

Advanced banking chatbot that retrieves relevant information from documents before generating responses.

## 🎯 What is RAG?

RAG combines:
1. **Document Retrieval**: Searches knowledge base for relevant information
2. **Context Injection**: Provides retrieved info to the LLM
3. **Grounded Response**: LLM generates answers based on actual documents

**Benefits:**
- ✅ Accurate answers from your documents
- ✅ Reduces hallucinations
- ✅ Cites sources
- ✅ Easy to update knowledge base

## 📦 Two Implementations

### 1. Basic RAG (`rag_banking_chatbot.py`)
- Simple vector similarity search
- Minimal dependencies
- Good for learning RAG concepts
- Built-in fallback embeddings

### 2. Advanced RAG (`advanced_rag_chatbot.py`)
- Uses ChromaDB vector database
- Better performance
- Persistent storage
- Production-ready

## 🚀 Quick Start

### Basic RAG
```bash
pip install requests numpy
python rag_banking_chatbot.py
```

### Advanced RAG (Recommended)
```bash
pip install -r requirements_rag.txt
python advanced_rag_chatbot.py
```

## 🔧 LLM Setup

### Ollama (Local - Recommended)

**Install Ollama:**
1. Download from https://ollama.ai
2. Install application

**Pull Models:**
```bash
# Chat model
ollama pull llama2

# Embedding model (for Basic RAG)
ollama pull nomic-embed-text

# Start server
ollama serve
```

### Hugging Face (Online)
No setup! Select option 2 when prompted.

## 📚 Knowledge Base

### Included Documents

The chatbot comes with sample banking documents in `knowledge_base/`:
- **savings.txt** - Savings account rates and features
- **loans.txt** - Personal, home, auto, education loans
- **credit_cards.txt** - Credit card products and benefits
- **policies.txt** - Banking policies, hours, limits
- **fixed_deposits.txt** - FD rates and terms

### Add Your Own Documents

**Method 1: Direct File Addition**
```bash
# Create .txt files in knowledge_base/ directory
echo "Insurance products information..." > knowledge_base/insurance.txt

# Restart chatbot to load new documents
python rag_banking_chatbot.py
```

**Method 2: Interactive Addition**
```
You: add
Filename: investment.txt
Content (type END when done):
Investment Products
Mutual Funds: Starting from $100
Bonds: Government and corporate
Stocks: Trade with zero commission
END

✓ Added investment.txt with 8 chunks
```

## 💬 Usage Examples

### Example 1: Basic Query
```
You: What's the interest rate on savings accounts?

🔍 Searching knowledge base...
✓ Found 3 relevant documents
  - savings.txt (similarity: 0.89)
  - fixed_deposits.txt (similarity: 0.72)
💭 Generating response...

======================================================================
Assistant: Our Savings Account offers a 4% interest rate per annum. 
For Premium Savings accounts with balances over $10,000, you can earn 
4.5% per annum. Minor accounts (under 18) receive 5% interest. Interest 
is calculated daily and credited quarterly.

📚 Sources: savings.txt
======================================================================
```

### Example 2: Complex Query
```
You: I need a loan for home renovation. What are my options?

🔍 Searching knowledge base...
✓ Found 3 relevant documents
  - loans.txt (similarity: 0.91)
  - policies.txt (similarity: 0.65)
💭 Generating response...

======================================================================
Assistant: For home renovation, you have two main options:

1. Personal Loan: 9-12% interest rate, amounts from $5,000 to $100,000, 
   tenure 1-5 years, with a 2% processing fee.

2. Home Loan: If you own the property, 8-10% interest rate, up to 
   $500,000, with tenure up to 30 years and 1% processing fee.

The home loan offers lower rates but requires property documentation. 
The personal loan is faster to approve (48 hours) and requires less 
documentation.

📚 Sources: loans.txt, policies.txt
======================================================================
```

### Example 3: Comparative Query
```
You: Compare your credit cards

🔍 Searching knowledge base...
✓ Found 2 relevant documents
  - credit_cards.txt (similarity: 0.94)
💭 Generating response...

======================================================================
Assistant: We have three credit card tiers:

Classic Card: $50 annual fee, 18% rate, $2k-$10k limit, 1 point/$100
Gold Card: $150 fee, 16% rate, $10k-$50k limit, 2 points/$100, 
           4 airport lounge visits/year
Platinum Card: $300 fee, 14% rate, $50k+ limit, 3 points/$100, 
               unlimited lounge access, concierge service

All cards include zero fraud liability, purchase protection, and 
EMI conversion facility.

📚 Sources: credit_cards.txt
======================================================================
```

## 🔍 How RAG Works

1. **User asks question** → "What's the FD rate for 3 years?"

2. **Embedding generation** → Question converted to vector embedding

3. **Similarity search** → Find most relevant document chunks
   ```
   ✓ fixed_deposits.txt (score: 0.91)
   ✓ savings.txt (score: 0.67)
   ```

4. **Context retrieval** → Extract relevant text from documents

5. **Prompt construction** → Combine context + question + instructions

6. **LLM generation** → Generate answer based on retrieved context

7. **Response with sources** → Return answer + cite sources

## 📊 RAG Architecture

```
User Question
    ↓
[Embedding Model]
    ↓
Query Vector
    ↓
[Vector Database Search]
    ↓
Retrieved Documents
    ↓
[Context + Question] → [LLM] → Response + Sources
```

## 🎛️ Commands

- `quit` or `exit` - Exit the chatbot
- `clear` - Clear conversation history
- `add` - Add new document to knowledge base

## ⚙️ Configuration

### Adjust Chunk Size

Edit in the code:
```python
def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50)
```

- `chunk_size`: Words per chunk (default: 500)
- `overlap`: Overlapping words between chunks (default: 50)

### Change Number of Retrieved Documents

```python
def retrieve_relevant_chunks(self, query: str, top_k: int = 3)
```

- `top_k`: Number of chunks to retrieve (default: 3)

### Switch LLM Models

**Ollama:**
```python
# In query_ollama method
payload = {
    "model": "mistral",  # Change to: llama2, mistral, phi, etc.
    "prompt": prompt,
    "stream": False
}
```

**Hugging Face:**
```python
# Change model URL
self.hf_api_url = "https://api-inference.huggingface.co/models/MODEL_NAME"
```

## 🔧 Troubleshooting

### ChromaDB Issues

**Error: "chromadb not found"**
```bash
pip install chromadb
```

**Error: "sqlite3 version too old"**
```bash
pip install pysqlite3-binary
```

### Ollama Issues

**Error: "Ollama not running"**
```bash
ollama serve
# Keep this terminal open
```

**Error: "Model not found"**
```bash
ollama pull llama2
ollama list  # Verify installation
```

### Embedding Issues

**Error: "nomic-embed-text not found"**
```bash
ollama pull nomic-embed-text
```

The basic RAG version has a fallback to simple embeddings if Ollama embeddings fail.

## 🚀 Advanced Features

### 1. Add Multiple Documents at Once

```bash
# Place all .txt files in knowledge_base/
cp /path/to/documents/*.txt knowledge_base/
python rag_banking_chatbot.py
```

### 2. Update Knowledge Base

Simply add new files to `knowledge_base/` and restart the chatbot.

### 3. Query Specific Sources

While the chatbot automatically finds relevant sources, you can ask:
```
You: What does the loan policy say about prepayment?
```

### 4. Persistent Storage (Advanced RAG)

ChromaDB persists data in `./chroma_db/` directory. Your knowledge base is saved between runs!

## 📈 Performance Tips

1. **Optimal Chunk Size**: 400-600 words works best for most documents
2. **Document Quality**: Clean, well-structured documents improve retrieval
3. **Specific Queries**: More specific questions get better results
4. **Model Selection**: Mistral generally performs better than LLaMA2 for banking
5. **Top-K Setting**: Use 3-5 chunks for best balance of context and speed

## 🔒 Security Notes

⚠️ **Important for Production:**
- Add authentication before deploying
- Sanitize user inputs
- Encrypt sensitive documents
- Implement access controls
- Add audit logging
- Use HTTPS for API calls

## 📝 Document Guidelines

For best RAG performance:

✅ **Do:**
- Use clear, structured content
- Include relevant keywords
- Break into logical sections
- Use consistent formatting
- Update regularly

❌ **Don't:**
- Use overly long paragraphs
- Include irrelevant information
- Mix unrelated topics in one document
- Use ambiguous language

## 🎓 Learning Resources

- **RAG Concepts**: https://python.langchain.com/docs/tutorials/rag/
- **Ollama Models**: https://ollama.ai/library
- **ChromaDB**: https://docs.trychroma.com/
- **Embeddings**: https://platform.openai.com/docs/guides/embeddings

## 🤝 Contributing

Enhance this chatbot by:
- Adding more document types (PDF, DOCX)
- Implementing metadata filtering
- Adding multi-language support
- Creating a web interface
- Integrating with databases

## 📄 License

Free to use and modify for educational and commercial purposes.

## 🆘 Support

For issues:
1. Check troubleshooting section
2. Verify all dependencies installed
3. Ensure Ollama is running (if using local)
4. Check document format in knowledge_base/

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Choose LLM provider (Ollama recommended)
3. ✅ Run the chatbot
4. ✅ Add your banking documents
5. ✅ Test with queries
6. ✅ Customize for your needs

Start with the basic RAG to understand concepts, then upgrade to advanced RAG for production use!
