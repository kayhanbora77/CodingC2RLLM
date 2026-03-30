"""
Advanced RAG Banking Chatbot with ChromaDB
Uses ChromaDB for efficient vector storage and retrieval
"""

import os
from typing import List, Dict
import requests

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not installed. Install with: pip install chromadb")


class AdvancedRAGChatbot:
    def __init__(self, model_provider="ollama", knowledge_base_dir="knowledge_base"):
        """Initialize advanced RAG chatbot with ChromaDB"""

        if not CHROMADB_AVAILABLE:
            raise ImportError("Please install chromadb: pip install chromadb")

        self.model_provider = model_provider
        self.knowledge_base_dir = knowledge_base_dir
        self.conversation_history = []

        os.makedirs(knowledge_base_dir, exist_ok=True)

        # Initialize ChromaDB
        print("Initializing ChromaDB...")
        self.chroma_client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory="./chroma_db",
            )
        )

        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(name="banking_docs")
            print(
                f"✓ Loaded existing collection with {self.collection.count()} documents"
            )
        except:
            self.collection = self.chroma_client.create_collection(
                name="banking_docs", metadata={"description": "Banking knowledge base"}
            )
            print("✓ Created new ChromaDB collection")
            self.load_knowledge_base()

        # API configurations
        self.ollama_url = "http://localhost:11434/api"
        self.hf_api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        self.hf_headers = {"Content-Type": "application/json"}

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def load_knowledge_base(self):
        """Load documents into ChromaDB"""
        print(f"Loading documents from {self.knowledge_base_dir}...")

        # Create sample documents if none exist
        from pathlib import Path

        txt_files = list(Path(self.knowledge_base_dir).glob("*.txt"))

        if not txt_files:
            print("Creating sample knowledge base...")
            self.create_sample_docs()
            txt_files = list(Path(self.knowledge_base_dir).glob("*.txt"))

        # Process each document
        doc_count = 0
        chunk_count = 0

        for filepath in txt_files:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self.chunk_text(content)

            # Add chunks to ChromaDB
            ids = [f"{filepath.stem}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {"filename": filepath.name, "chunk_id": i} for i in range(len(chunks))
            ]

            self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)

            doc_count += 1
            chunk_count += len(chunks)
            print(f"  ✓ {filepath.name}: {len(chunks)} chunks")

        print(f"✓ Loaded {chunk_count} chunks from {doc_count} documents")

    def create_sample_docs(self):
        """Create sample banking documents"""
        docs = {
            "savings.txt": """Savings Account: 4% interest, $500 min balance, free ATM withdrawals.
Premium Savings (>$10k): 4.5% interest. Minor accounts: 5% interest.
Features: Online banking, debit card, overdraft protection.
Open account: Visit branch with ID, fill form, deposit $500.""",
            "loans.txt": """Personal Loans: 9-12% rate, $5k-$100k, 1-5 years, 2% processing fee.
Home Loans: 8-10% rate, up to $500k, 30 years max, 1% processing fee.
Auto Loans: 10-13% rate, up to $75k, 1-7 years.
Education Loans: 8-9% rate, no collateral up to $50k.""",
            "credit_cards.txt": """Classic Card: $50 annual fee, 18% rate, $2k-$10k limit, 1 point per $100.
Gold Card: $150 fee, 16% rate, $10k-$50k limit, 2 points per $100, airport lounge.
Platinum Card: $300 fee, 14% rate, $50k+ limit, 3 points per $100, unlimited lounge.
All cards: Zero fraud liability, purchase protection, EMI facility.""",
            "policies.txt": """Hours: Mon-Fri 9AM-5PM, Sat 9AM-1PM. Closed Sunday.
Customer Service: 1-800-BANK-HELP, email: support@ourbank.com
Daily limits: ATM $1,000, online transfer $10,000, international $5,000.
Security: 2FA, SMS alerts, biometric login. Complaints resolved in 7 days.""",
            "fixed_deposits.txt": """FD Rates: 3mo 5.5%, 6mo 6%, 1yr 6.5%, 2yr 6.8%, 3yr 7%, 5yr 7.2%.
Senior citizens: +0.5% interest. Min deposit: $1,000, no max.
Features: Premature withdrawal (1% penalty), loan against FD (90% value).
5-year FD: Tax deduction under Section 80C.""",
        }

        for filename, content in docs.items():
            with open(os.path.join(self.knowledge_base_dir, filename), "w") as f:
                f.write(content)

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant context from ChromaDB"""
        results = self.collection.query(query_texts=[query], n_results=top_k)

        retrieved = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                retrieved.append(
                    {
                        "text": doc,
                        "filename": results["metadatas"][0][i]["filename"],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else 0,
                    }
                )

        return retrieved

    def query_llm(self, prompt: str) -> str:
        """Query LLM (Ollama or HuggingFace)"""
        if self.model_provider == "ollama":
            try:
                response = requests.post(
                    f"{self.ollama_url}/generate",
                    json={"model": "llama2", "prompt": prompt, "stream": False},
                    timeout=60,
                )
                if response.status_code == 200:
                    return response.json()["response"]
                return "Error: Ollama connection failed"
            except:
                return "Error: Ollama not running"
        else:
            try:
                response = requests.post(
                    self.hf_api_url,
                    headers=self.hf_headers,
                    json={
                        "inputs": prompt,
                        "parameters": {"max_new_tokens": 400, "temperature": 0.7},
                    },
                    timeout=60,
                )
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and result:
                        return result[0].get("generated_text", "No response")
                return "Error: HuggingFace API failed"
            except:
                return "Error: HuggingFace connection failed"

    def get_response(self, question: str) -> Dict:
        """Get RAG-enhanced response"""

        # Retrieve context
        print("🔍 Searching knowledge base...")
        context_chunks = self.retrieve_context(question, top_k=3)

        if context_chunks:
            print(f"✓ Found {len(context_chunks)} relevant sections")
            for chunk in context_chunks:
                print(f"  - {chunk['filename']}")

        # Build prompt
        context_text = "\n\n".join(
            [f"[{c['filename']}]: {c['text']}" for c in context_chunks]
        )

        prompt = f"""You are a banking assistant. Use this context to answer:

Context:
{context_text}

Question: {question}

Provide a helpful, accurate answer based on the context. Be concise.

Answer:"""

        # Get LLM response
        print("💭 Generating response...")
        response = self.query_llm(prompt)

        # Store history
        self.conversation_history.append({"user": question, "assistant": response})

        return {
            "response": response,
            "sources": [c["filename"] for c in context_chunks],
        }

    def add_document(self, filename: str, content: str):
        """Add new document to knowledge base"""
        filepath = os.path.join(self.knowledge_base_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)

        chunks = self.chunk_text(content)
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"filename": filename, "chunk_id": i} for i in range(len(chunks))]

        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)

        print(f"✓ Added {filename} with {len(chunks)} chunks")

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✓ History cleared")


def main():
    """Main function"""
    print("=" * 70)
    print("Advanced RAG Banking Chatbot with ChromaDB")
    print("=" * 70)

    print("\nChoose LLM:")
    print("1. Ollama (Local)")
    print("2. Hugging Face (Online)")

    choice = input("\nChoice (1/2): ").strip()
    provider = "ollama" if choice == "1" else "huggingface"

    if provider == "ollama":
        print("\n📌 Ollama: Ensure 'ollama serve' and 'ollama pull llama2'")
    else:
        print("\n📌 HuggingFace: May be slow on first request")

    try:
        chatbot = AdvancedRAGChatbot(model_provider=provider)
    except ImportError as e:
        print(f"\n❌ {e}")
        return

    print("\n" + "=" * 70)
    print("Ready! Commands: 'quit', 'clear', 'add'")
    print("=" * 70 + "\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("\nGoodbye!")
            break

        if user_input.lower() == "clear":
            chatbot.clear_history()
            continue

        if user_input.lower() == "add":
            filename = input("Filename: ").strip()
            print("Content (type END when done):")
            lines = []
            while True:
                line = input()
                if line == "END":
                    break
                lines.append(line)
            chatbot.add_document(filename, "\n".join(lines))
            continue

        print()
        result = chatbot.get_response(user_input)

        print("\n" + "=" * 70)
        print("Assistant:", result["response"])
        if result["sources"]:
            print(f"\n📚 Sources: {', '.join(set(result['sources']))}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
