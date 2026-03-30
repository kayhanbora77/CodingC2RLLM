# Email Auto-Reply System with RAG
### Optimized for Lenovo G580 / 12 GB RAM / No GPU

---

## What this system does

1. Watches your inbox every 60 seconds (IMAP)
2. When a new email arrives, finds the most relevant FAQ answers using semantic search (RAG)
3. Passes those answers + the email to a tiny local LLM to write a polished reply
4. Shows you the draft and asks for confirmation before sending (or auto-sends if you prefer)

---

## Hardware-aware choices

| Component | Choice | Why it fits your laptop |
|-----------|--------|------------------------|
| LLM | TinyLlama 1.1B (via Ollama) | Only 637 MB, runs on CPU in ~10 sec/reply |
| Embeddings | all-MiniLM-L6-v2 | Only 80 MB, very fast on CPU |
| Vector DB | ChromaDB (in-process) | No server, runs in RAM (~50 MB) |
| Total RAM needed | ~1.5 GB | Leaves plenty of your 12 GB free |

---

## Step 1 — Install Ollama and pull TinyLlama

Ollama runs the LLM locally. It's free, no API key needed.

```bash
# Download from: https://ollama.com/download
# Then open a terminal and run:
ollama pull tinyllama
```

> **Keep Ollama running** in the background while using this system.
> It listens on `http://localhost:11434`

If your laptop feels slow, try even lighter models:
- `ollama pull tinyllama`   → 637 MB  ✅ recommended
- `ollama pull phi`         → 1.6 GB  (smarter but slower)
- `ollama pull gemma:2b`    → 1.7 GB  (good quality)

---

## Step 2 — Install Python dependencies

```bash
pip install chromadb sentence-transformers python-dotenv colorama imaplib2 requests
```

> First run downloads the embedding model (~80 MB). After that it's cached locally.

---

## Step 3 — Configure your credentials

Copy `.env.example` to `.env` and fill in your details:

```bash
cp .env.example .env
```

Edit `.env`:
```
EMAIL_ADDRESS=you@gmail.com
EMAIL_PASSWORD=xxxx xxxx xxxx xxxx   ← Gmail App Password (NOT your real password)
```

### How to get a Gmail App Password
1. Go to https://myaccount.google.com/security
2. Enable **2-Step Verification** if not already on
3. Search for "App Passwords"
4. Create one for "Mail" → copy the 16-character password
5. Paste it into `.env` as `EMAIL_PASSWORD`

---

## Step 4 — Put your FAQ CSV in the same folder

Make sure `click2refund_faq.csv` is in the same folder as `main.py`.

---

## Step 5 — Test the RAG without email first

```bash
python test_rag.py
```

Type test questions like:
- "my flight was delayed 4 hours, can I get money?"
- "how long does a claim take?"
- "do I need to pay anything upfront?"

You'll see the top matching FAQ entries with similarity scores.

---

## Step 6 — Run the auto-reply system

```bash
python main.py
```

By default it will **show you each draft reply** and ask `Send? [y/N]` before sending.

To enable fully automatic sending, set in `.env`:
```
AUTO_SEND=true
```

---

## File structure

```
email_rag_reply/
├── main.py              ← main auto-reply loop
├── test_rag.py          ← test RAG without email
├── requirements.txt     ← Python packages
├── .env.example         ← copy to .env and fill in
└── click2refund_faq.csv ← your FAQ data
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on Ollama | Run `ollama serve` in a terminal first |
| Gmail login fails | Use App Password, not your real password |
| Replies are off-topic | Lower `CONFIDENCE_THRESHOLD` to 0.35 |
| System is slow | Switch to `tinyllama` model in `.env` |
| Out of memory | Close browser tabs, set `CHECK_INTERVAL_SECONDS=120` |

---

## RAM usage summary

```
Ollama (TinyLlama) .............. ~800 MB
Embedding model (MiniLM) ........ ~200 MB
ChromaDB (50 FAQ vectors) ....... ~50 MB
Python + libraries .............. ~300 MB
─────────────────────────────────────────
Total ........................... ~1.4 GB  (out of your 12 GB)
```

You can safely run this alongside a browser and other apps.
