# banking_chatbot_rag.py
import os
import time
import random
from typing import List

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ────────────────────────────────────────────────
#               CONFIGURATION
# ────────────────────────────────────────────────

OLLAMA_MODEL = "llama3.2:3b"  # or "phi4:mini", "gemma2:2b", "qwen2.5:3b"
EMBEDDING_MODEL = "mxbai-embed-large"  # very good small model in 2026
# Alternative: "nomic-embed-text", "bge-m3" (a bit heavier)

DOCS_FOLDER = "./bank_docs"
CHROMA_DIR = "./chroma_db_banking"

# ────────────────────────────────────────────────
#             LOAD & INDEX DOCUMENTS
# ────────────────────────────────────────────────


def load_and_index_documents():
    if os.path.exists(CHROMA_DIR):
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        )
        return vectorstore

    print("Creating new vector store...")

    # Support both PDF and TXT
    loader = DirectoryLoader(
        DOCS_FOLDER,
        glob="**/*",
        loader_cls=PyPDFLoader,  # default for pdf
        recursive=True,
        silent_errors=True,
    )
    docs = loader.load()

    # Add txt files if PyPDFLoader skipped them
    for file in os.listdir(DOCS_FOLDER):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCS_FOLDER, file))
            docs.extend(loader.load())

    if not docs:
        raise ValueError("No documents found in bank_docs folder!")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=120, separators=["\n\n", "\n", ".", " ", ""]
    )

    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks from documents.")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DIR,
    )

    print("Vector store created and saved.")
    return vectorstore


# ────────────────────────────────────────────────
#                 RAG CHAIN SETUP
# ────────────────────────────────────────────────


def create_rag_chain(vectorstore):
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1, num_predict=350)

    # Good prompt for banking context - strict, safe, no hallucination
    prompt_template = """You are a polite, accurate and very careful customer service assistant of YourBank.
You ONLY answer based on the provided documents and official bank knowledge.
If the information is not in the documents or you are not 100% sure, say:
"I'm sorry, I don't have that information. Please contact customer care at 1800-123-4567."

Never guess. Never give financial advice. Never ask for PIN, OTP, password, CVV.

Context (most relevant parts):
{context}

Question: {question}

Answer concisely, clearly and safely:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return qa_chain


# ────────────────────────────────────────────────
#                 MAIN CHATBOT
# ────────────────────────────────────────────────

GREETING_RESPONSES = [
    "Hello! Welcome to YourBank AI Assistant 🏦 How may I assist you today?",
    "Hi! I'm here to help with your banking queries. What can I do for you?",
]

SORRY_RESPONSES = [
    "I'm sorry, I don't have enough information to answer that.",
    "That question is outside my current knowledge base. Please call 1800-123-4567.",
]


def main():
    print("\n" + "═" * 60)
    print("   YourBank RAG-powered Chatbot   (Feb 2026 edition)  🏦")
    print("═" * 60)
    print("Type 'exit', 'quit' or 'bye' to end\n")

    vectorstore = load_and_index_documents()
    rag_chain = create_rag_chain(vectorstore)

    print(random.choice(GREETING_RESPONSES))
    print()

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "goodbye", "q"]:
                print("\nThank you for banking with us. Have a great day! 👋")
                break

            if not user_input:
                continue

            print("Thinking...", end="", flush=True)

            result = rag_chain.invoke({"query": user_input})

            answer = result["result"].strip()
            sources = result["source_documents"]

            # Clean up loading message
            print("\r" + " " * 12 + "\r", end="")

            print(f"BankBot: {answer}")

            # Optional: show sources (good for debugging & trust)
            if sources:
                print("\nSources:")
                for i, doc in enumerate(sources, 1):
                    src = doc.metadata.get("source", "unknown").split("/")[-1]
                    print(f"  {i}. {src}")

            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("BankBot: Sorry, something went wrong. Please try again.\n")


if __name__ == "__main__":
    print("Make sure Ollama is running with:")
    print(f"  ollama run {OLLAMA_MODEL}")
    print(f"  ollama pull {EMBEDDING_MODEL}\n")

    main()
