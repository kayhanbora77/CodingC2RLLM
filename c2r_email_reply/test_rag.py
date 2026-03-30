"""
test_rag.py  –  Test the RAG pipeline without email / Ollama
Run: python test_rag.py
"""

import csv, os
import chromadb
from sentence_transformers import SentenceTransformer
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

FAQ_CSV = os.path.join(os.path.dirname(__file__), "click2refund_faq.csv")
EMBED   = "all-MiniLM-L6-v2"

# ── Load FAQ ────────────────────────────────────────────────────
faqs = []
with open(FAQ_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        q = row.get("Question", "").strip()
        a = row.get("Answer", "").strip()
        if q and a:
            faqs.append({"question": q, "answer": a})
print(Fore.GREEN + f"✔ Loaded {len(faqs)} FAQ entries")

if not faqs:
    print(Fore.RED + "✘ No FAQ entries found. Check that click2refund_faq.csv has Question and Answer columns with data.")
    exit(1)

# ── Build vector store ──────────────────────────────────────────
print(Fore.YELLOW + f"▶ Loading embeddings model (downloads ~80MB on first run)...")
model = SentenceTransformer(EMBED)

client     = chromadb.Client()
collection = client.get_or_create_collection("faq_test")

docs, metas, ids = [], [], []
for i, faq in enumerate(faqs):
    docs.append(faq["question"] + " " + faq["answer"])
    metas.append({"question": faq["question"], "answer": faq["answer"]})
    ids.append(str(i))

embs = model.encode(docs, show_progress_bar=True).tolist()
collection.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
print(Fore.GREEN + "✔ Vector store ready\n")

# ── Interactive query loop ──────────────────────────────────────
print(Fore.CYAN + "─── Interactive RAG Test ─── (type 'quit' to exit)\n")
while True:
    query = input(Fore.WHITE + "Enter a customer question:\n> ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    if not query:
        continue

    q_emb   = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)

    print()
    for i in range(len(results["ids"][0])):
        meta  = results["metadatas"][0][i]
        dist  = results["distances"][0][i]
        score = round(1 - dist, 3)
        color = Fore.GREEN if score >= 0.45 else Fore.RED
        print(color + f"  [{score:.3f}]  Q: {meta['question']}")
        print(f"          A: {meta['answer'][:160]}...")
        print()
