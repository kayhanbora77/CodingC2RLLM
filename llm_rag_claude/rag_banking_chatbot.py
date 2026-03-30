"""
Banking Chatbot with RAG (Retrieval-Augmented Generation)
Uses vector embeddings to retrieve relevant information from documents
"""

import json
import os
from typing import List, Dict, Optional
import requests
import numpy as np
from pathlib import Path


class RAGBankingChatbot:
    def __init__(self, model_provider="ollama", knowledge_base_dir="knowledge_base"):
        """
        Initialize RAG-enabled banking chatbot

        Args:
            model_provider: "ollama" or "huggingface"
            knowledge_base_dir: Directory containing banking documents
        """
        self.model_provider = model_provider
        self.knowledge_base_dir = knowledge_base_dir
        self.conversation_history = []

        # Create knowledge base directory if it doesn't exist
        os.makedirs(knowledge_base_dir, exist_ok=True)

        # Vector store for embeddings
        self.document_chunks = []
        self.embeddings = []

        # API configurations
        self.ollama_url = "http://localhost:11434/api"
        self.hf_api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        self.hf_embed_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.hf_headers = {"Content-Type": "application/json"}

        print("Initializing RAG system...")
        self.load_knowledge_base()

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def load_documents_from_directory(self) -> List[Dict[str, str]]:
        """Load all documents from knowledge base directory"""
        documents = []

        # Check for text files
        for file_path in Path(self.knowledge_base_dir).glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({"filename": file_path.name, "content": content})

        return documents

    def get_embedding_ollama(self, text: str) -> List[float]:
        """Get embedding using Ollama"""
        try:
            url = f"{self.ollama_url}/embeddings"
            payload = {
                "model": "nomic-embed-text",  # Ollama embedding model
                "prompt": text,
            }

            response = requests.post(url, json=payload, timeout=30)

            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                # Fallback: Simple word count embedding (basic)
                return self.simple_embedding(text)

        except:
            return self.simple_embedding(text)

    def get_embedding_huggingface(self, text: str) -> List[float]:
        """Get embedding using Hugging Face"""
        try:
            payload = {"inputs": text}
            response = requests.post(
                self.hf_embed_url, headers=self.hf_headers, json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list):
                    return result
                return result.get("embeddings", self.simple_embedding(text))
            else:
                return self.simple_embedding(text)

        except:
            return self.simple_embedding(text)

    def simple_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Simple embedding fallback using term frequency"""
        words = text.lower().split()
        word_freq = {}

        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Create a simple vector
        embedding = [0.0] * dim
        for i, word in enumerate(list(word_freq.keys())[:dim]):
            embedding[i] = word_freq[word] / len(words)

        # Normalize
        norm = sum(x**2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding based on provider"""
        if self.model_provider == "ollama":
            return self.get_embedding_ollama(text)
        else:
            return self.get_embedding_huggingface(text)

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def load_knowledge_base(self):
        """Load and process all documents into vector store"""
        print(f"Loading documents from {self.knowledge_base_dir}...")

        documents = self.load_documents_from_directory()

        if not documents:
            print("No documents found. Creating sample knowledge base...")
            self.create_sample_knowledge_base()
            documents = self.load_documents_from_directory()

        # Process each document
        for doc in documents:
            chunks = self.chunk_text(doc["content"])
            print(f"Processing {doc['filename']}: {len(chunks)} chunks")

            for chunk in chunks:
                embedding = self.get_embedding(chunk)
                self.document_chunks.append(
                    {"text": chunk, "filename": doc["filename"], "embedding": embedding}
                )

        print(
            f"✓ Loaded {len(self.document_chunks)} chunks from {len(documents)} documents"
        )

    def create_sample_knowledge_base(self):
        """Create sample banking documents"""
        documents = {
            "savings_accounts.txt": """
Savings Account Information

Our Savings Account is designed to help you grow your money with competitive interest rates.

Interest Rates:
- Regular Savings: 4.0% per annum
- Premium Savings (balance > $10,000): 4.5% per annum
- Minor Savings (under 18): 5.0% per annum

Features:
- Minimum balance: $500
- Free ATM withdrawals: 5 per month
- Additional ATM withdrawal fee: $2 per transaction
- Monthly statement provided
- Online banking and mobile app access
- Debit card included
- Overdraft protection available

To open a savings account:
1. Visit any branch with valid ID
2. Fill out account opening form
3. Make initial deposit of minimum $500
4. Receive account number and debit card within 5 business days

Interest is calculated daily and credited quarterly.
            """,
            "loans.txt": """
Loan Products and Services

Personal Loans:
- Interest Rate: 9-12% per annum (based on credit score)
- Loan Amount: $5,000 to $100,000
- Tenure: 1 to 5 years
- Processing Fee: 2% of loan amount
- No prepayment penalty
- Approval time: 48 hours
- Required documents: ID proof, income proof, address proof

Home Loans:
- Interest Rate: 8-10% per annum
- Loan Amount: Up to $500,000
- Tenure: Up to 30 years
- Loan-to-Value ratio: Up to 85%
- Processing Fee: 1% of loan amount
- Tax benefits available under Section 80C and 24(b)
- Required documents: Property papers, income proof, ID proof

Auto Loans:
- Interest Rate: 10-13% per annum
- Loan Amount: Up to $75,000
- Tenure: 1 to 7 years
- Finance up to 90% of vehicle value
- Quick approval within 24 hours

Education Loans:
- Interest Rate: 8-9% per annum
- Cover tuition fees, living expenses, books
- Moratorium period during study + 1 year
- No collateral for loans up to $50,000
            """,
            "credit_cards.txt": """
Credit Card Services

Classic Credit Card:
- Annual Fee: $50 (waived on spending $5,000 annually)
- Interest Rate: 18% per annum on outstanding balance
- Credit Limit: $2,000 to $10,000
- Reward Points: 1 point per $100 spent
- Interest-free period: 45 days

Gold Credit Card:
- Annual Fee: $150 (waived on spending $15,000 annually)
- Interest Rate: 16% per annum
- Credit Limit: $10,000 to $50,000
- Reward Points: 2 points per $100 spent
- Travel insurance included
- Airport lounge access: 4 visits per year

Platinum Credit Card:
- Annual Fee: $300
- Interest Rate: 14% per annum
- Credit Limit: $50,000 and above
- Reward Points: 3 points per $100 spent
- Comprehensive travel insurance
- Unlimited airport lounge access
- Concierge service

All cards include:
- Zero fraud liability
- Purchase protection
- EMI conversion facility
- Balance transfer option
- Contactless payment enabled
            """,
            "policies.txt": """
Banking Policies and Procedures

Operating Hours:
- Monday to Friday: 9:00 AM to 5:00 PM
- Saturday: 9:00 AM to 1:00 PM
- Sunday and Public Holidays: Closed
- ATMs: 24/7 availability

Customer Service:
- Phone: 1-800-BANK-HELP (1-800-2265-4357)
- Email: support@ourbank.com
- Live Chat: Available on website 24/7
- Response time: Within 24 hours

Transaction Limits:
- Daily ATM withdrawal: $1,000
- Daily online transfer: $10,000
- International transaction: $5,000
- Contact us to increase limits

Security Measures:
- Two-factor authentication for online banking
- SMS alerts for all transactions
- Card blocking facility via phone/app
- Biometric login available on mobile app

Complaint Resolution:
- File complaint online or at branch
- Acknowledgment within 24 hours
- Resolution within 7 working days
- Escalation to manager if unresolved

Account Closure:
- Submit written request at branch
- Clear all outstanding dues
- Return debit/credit cards
- Final statement provided
- Process completed within 14 days
            """,
            "fixed_deposits.txt": """
Fixed Deposit (FD) Information

Fixed Deposits offer secure investment with guaranteed returns.

Interest Rates:
- 3 months: 5.5% per annum
- 6 months: 6.0% per annum
- 1 year: 6.5% per annum
- 2 years: 6.8% per annum
- 3 years: 7.0% per annum
- 5 years: 7.2% per annum

Senior Citizens get 0.5% additional interest on all tenures.

Features:
- Minimum deposit: $1,000
- Maximum deposit: No limit
- Interest payout: Monthly, quarterly, annual, or on maturity
- Premature withdrawal allowed with penalty of 1%
- Loan against FD: Up to 90% of deposit amount at 2% above FD rate
- Auto-renewal facility available
- Nomination facility available

Tax Benefits:
- 5-year tax-saving FD eligible for deduction under Section 80C
- TDS deducted if interest exceeds $10,000 annually
- Submit Form 15G/15H to avoid TDS if income below taxable limit

To open FD:
1. Visit branch or use online banking
2. Choose amount and tenure
3. Select interest payout frequency
4. FD receipt issued immediately
            """,
        }

        for filename, content in documents.items():
            filepath = os.path.join(self.knowledge_base_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

        print(
            f"✓ Created {len(documents)} sample documents in {self.knowledge_base_dir}/"
        )

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        if not self.document_chunks:
            return []

        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Calculate similarities
        similarities = []
        for i, chunk in enumerate(self.document_chunks):
            similarity = self.cosine_similarity(query_embedding, chunk["embedding"])
            similarities.append((i, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top K chunks
        results = []
        for i, score in similarities[:top_k]:
            results.append(
                {
                    "text": self.document_chunks[i]["text"],
                    "filename": self.document_chunks[i]["filename"],
                    "score": score,
                }
            )

        return results

    def query_ollama(self, prompt: str) -> str:
        """Query Ollama model"""
        try:
            url = f"{self.ollama_url}/generate"
            payload = {"model": "llama2", "prompt": prompt, "stream": False}

            response = requests.post(url, json=payload, timeout=60)

            if response.status_code == 200:
                return response.json()["response"]
            return f"Error: Status {response.status_code}"

        except requests.exceptions.ConnectionError:
            return "Error: Ollama not running. Start with 'ollama serve'"
        except Exception as e:
            return f"Error: {str(e)}"

    def query_huggingface(self, prompt: str) -> str:
        """Query Hugging Face model"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 400,
                    "temperature": 0.7,
                    "return_full_text": False,
                },
            }

            response = requests.post(
                self.hf_api_url, headers=self.hf_headers, json=payload, timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "No response")
                return str(result)
            elif response.status_code == 503:
                return "Model loading. Please retry."
            return f"Error: Status {response.status_code}"

        except Exception as e:
            return f"Error: {str(e)}"

    def create_rag_prompt(
        self, user_question: str, retrieved_chunks: List[Dict]
    ) -> str:
        """Create prompt with retrieved context"""

        # Build context from retrieved chunks
        context = "Relevant information from our knowledge base:\n\n"
        for i, chunk in enumerate(retrieved_chunks, 1):
            context += f"[Source {i} - {chunk['filename']}]:\n{chunk['text']}\n\n"

        # Add conversation history
        history = ""
        if self.conversation_history:
            history = "Recent conversation:\n"
            for ex in self.conversation_history[-2:]:
                history += f"User: {ex['user']}\nAssistant: {ex['assistant']}\n"

        prompt = f"""You are a helpful banking assistant. Use the provided context to answer questions accurately.

{context}

{history}

Customer Question: {user_question}

Instructions:
- Answer based on the context provided above
- Be accurate and cite information from the sources
- If the context doesn't contain the answer, say so and provide general guidance
- Keep response concise and professional

Assistant Response:"""

        return prompt

    def get_response(self, user_question: str) -> Dict:
        """Get RAG-enhanced response"""

        # Retrieve relevant chunks
        print("🔍 Searching knowledge base...")
        relevant_chunks = self.retrieve_relevant_chunks(user_question, top_k=3)

        if relevant_chunks:
            print(f"✓ Found {len(relevant_chunks)} relevant documents")
            for chunk in relevant_chunks:
                print(f"  - {chunk['filename']} (similarity: {chunk['score']:.2f})")
        else:
            print("⚠ No relevant documents found")

        # Create RAG prompt
        prompt = self.create_rag_prompt(user_question, relevant_chunks)

        # Query LLM
        print("💭 Generating response...")
        if self.model_provider == "ollama":
            response = self.query_ollama(prompt)
        else:
            response = self.query_huggingface(prompt)

        # Store in history
        self.conversation_history.append({"user": user_question, "assistant": response})

        return {
            "response": response,
            "sources": [chunk["filename"] for chunk in relevant_chunks],
            "retrieved_chunks": relevant_chunks,
        }

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✓ Conversation history cleared")

    def add_document(self, filename: str, content: str):
        """Add a new document to the knowledge base"""
        filepath = os.path.join(self.knowledge_base_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # Process and add to vector store
        chunks = self.chunk_text(content)
        for chunk in chunks:
            embedding = self.get_embedding(chunk)
            self.document_chunks.append(
                {"text": chunk, "filename": filename, "embedding": embedding}
            )

        print(f"✓ Added {filename} with {len(chunks)} chunks")


def main():
    """Main function"""
    print("=" * 70)
    print("Banking Chatbot with RAG (Retrieval-Augmented Generation)")
    print("=" * 70)

    print("\nChoose LLM provider:")
    print("1. Ollama (Local)")
    print("2. Hugging Face (Online)")

    choice = input("\nChoice (1/2): ").strip()
    provider = "ollama" if choice == "1" else "huggingface"

    if provider == "ollama":
        print("\n📌 Using Ollama")
        print("Requirements:")
        print("  - ollama serve (running)")
        print("  - ollama pull llama2")
        print("  - ollama pull nomic-embed-text (for embeddings)")
    else:
        print("\n📌 Using Hugging Face")
        print("Note: First request may be slow")

    # Initialize RAG chatbot
    chatbot = RAGBankingChatbot(model_provider=provider)

    print("\n" + "=" * 70)
    print("RAG Chatbot Ready!")
    print("The chatbot will search documents before answering your questions.")
    print("Commands: 'quit', 'clear', 'add' (to add document)")
    print("=" * 70 + "\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nThank you! Goodbye!")
            break

        if user_input.lower() == "clear":
            chatbot.clear_history()
            continue

        if user_input.lower() == "add":
            print("\nAdd new document to knowledge base:")
            filename = input("Filename (e.g., insurance.txt): ").strip()
            print("Enter document content (type 'END' on a new line when done):")
            lines = []
            while True:
                line = input()
                if line == "END":
                    break
                lines.append(line)
            content = "\n".join(lines)
            chatbot.add_document(filename, content)
            continue

        # Get RAG response
        print()
        result = chatbot.get_response(user_input)

        print("\n" + "=" * 70)
        print("Assistant:", result["response"])
        if result["sources"]:
            print(f"\n📚 Sources: {', '.join(set(result['sources']))}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
