"""
Email Auto-Reply System with RAG  –  Click2Refund FAQ
======================================================
Stack (CPU-friendly for Lenovo G580 / 12GB RAM):
  • Embeddings : sentence-transformers/all-MiniLM-L6-v2   (~80 MB)
  • Vector DB  : ChromaDB  (in-process, no server)
  • LLM        : TinyLlama via Ollama  (~637 MB)
  • Email      : IMAP (read) + SMTP (send)
"""

import os, csv, time, imaplib, smtplib, email, textwrap
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from datetime import datetime

import chromadb
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

# ── Init ────────────────────────────────────────────────────────
load_dotenv()
colorama_init(autoreset=True)

EMAIL_ADDRESS        = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD       = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER          = os.getenv("IMAP_SERVER", "imap.gmail.com")
SMTP_SERVER          = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT            = int(os.getenv("SMTP_PORT", 587))
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL", "tinyllama")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAQ_CSV_PATH_RAW    = os.getenv("FAQ_CSV_PATH", "click2refund_faq.csv")
# Always resolve relative to script if not absolute
if not os.path.isabs(FAQ_CSV_PATH_RAW):
    FAQ_CSV_PATH = os.path.join(os.path.dirname(__file__), FAQ_CSV_PATH_RAW)
else:
    FAQ_CSV_PATH = FAQ_CSV_PATH_RAW
CHECK_INTERVAL       = int(os.getenv("CHECK_INTERVAL_SECONDS", 60))
AUTO_SEND            = os.getenv("AUTO_SEND", "false").lower() == "true"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.45))

# ── 1. Load FAQ CSV ─────────────────────────────────────────────
def load_faq(path: str) -> list[dict]:
    faqs = []
    if not os.path.exists(path):
        print(Fore.RED + f"✘ FAQ file not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row.get("Question", "").strip()
            a = row.get("Answer", "").strip()
            if q and a:
                faqs.append({"question": q, "answer": a})
    
    if not faqs:
        print(Fore.RED + f"✘ Loaded 0 FAQ entries from {path}. Check CSV columns and data.")
        return []

    print(Fore.GREEN + f"✔ Loaded {len(faqs)} FAQ entries from {path}")
    return faqs

# ── 2. Build ChromaDB vector store ─────────────────────────────
def build_vector_store(faqs: list[dict], embed_model: SentenceTransformer):
    client = chromadb.Client()               # in-memory, no disk needed
    collection = client.get_or_create_collection("faq")

    documents, metadatas, ids = [], [], []
    for i, faq in enumerate(faqs):
        # Index question + answer together for richer retrieval
        documents.append(faq["question"] + " " + faq["answer"])
        metadatas.append({"question": faq["question"], "answer": faq["answer"]})
        ids.append(str(i))

    embeddings = embed_model.encode(documents, show_progress_bar=True).tolist()
    collection.add(documents=documents, embeddings=embeddings,
                   metadatas=metadatas, ids=ids)
    print(Fore.GREEN + f"✔ Vector store ready ({len(documents)} vectors)")
    return collection

# ── 3. RAG retrieval ────────────────────────────────────────────
def retrieve(query: str, collection, embed_model: SentenceTransformer,
             top_k: int = 3) -> list[dict]:
    q_emb = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    hits = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]          # lower = more similar
        similarity = 1 - dist                      # convert to similarity
        hits.append({
            "question": meta["question"],
            "answer":   meta["answer"],
            "score":    round(similarity, 3)
        })
    return hits

# ── 4. Generate reply via Ollama (TinyLlama) ────────────────────
def generate_reply(customer_email: str, context_hits: list[dict]) -> str:
    # Build context string from top FAQ hits
    context_parts = []
    for h in context_hits:
        if h["score"] >= CONFIDENCE_THRESHOLD:
            context_parts.append(f"Q: {h['question']}\nA: {h['answer']}")

    if not context_parts:
        return (
            "Thank you for contacting Click2Refund. "
            "One of our team members will review your message and get back to you shortly. "
            "For urgent matters, please visit https://www.click2refund.com/en/Help"
        )

    context = "\n\n".join(context_parts)

    prompt = textwrap.dedent(f"""
    You are a helpful customer support agent for Click2Refund, a flight compensation company.
    Use ONLY the FAQ context below to answer the customer's email.
    Be concise, friendly, and professional. Do not invent information.
    End with: "Best regards, Click2Refund Support Team"

    --- FAQ CONTEXT ---
    {context}

    --- CUSTOMER EMAIL ---
    {customer_email}

    --- YOUR REPLY ---
    """).strip()

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(Fore.RED + f"  LLM error: {e}")
        return None

# ── 5. Email utilities ──────────────────────────────────────────
def decode_str(s):
    parts = decode_header(s)
    decoded = []
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)

def fetch_unread_emails(imap: imaplib.IMAP4_SSL) -> list[dict]:
    imap.select("INBOX")
    _, data = imap.search(None, "UNSEEN")
    uids = data[0].split()
    emails = []
    for uid in uids:
        _, msg_data = imap.fetch(uid, "(RFC822)")
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)
        subject = decode_str(msg.get("Subject", "(no subject)"))
        sender  = decode_str(msg.get("From", ""))
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
        emails.append({"uid": uid, "sender": sender,
                       "subject": subject, "body": body.strip()})
    return emails

def send_reply(to_addr: str, subject: str, body: str):
    msg = MIMEMultipart()
    msg["From"]    = EMAIL_ADDRESS
    msg["To"]      = to_addr
    msg["Subject"] = f"Re: {subject}" if not subject.startswith("Re:") else subject
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_addr, msg.as_string())

def mark_as_read(imap: imaplib.IMAP4_SSL, uid):
    imap.store(uid, "+FLAGS", "\\Seen")

# ── 6. Main loop ────────────────────────────────────────────────
def main():
    print(Fore.CYAN + "\n╔══════════════════════════════════════════╗")
    print(Fore.CYAN + "║  Click2Refund Email Auto-Reply (RAG)    ║")
    print(Fore.CYAN + "╚══════════════════════════════════════════╝\n")

    # Load FAQ & build vector store
    faqs = load_faq(FAQ_CSV_PATH)
    if not faqs:
        print(Fore.RED + "Exiting due to empty FAQ database.")
        return

    print(Fore.YELLOW + f"▶ Loading embedding model: {EMBEDDING_MODEL}  (first run downloads ~80MB)...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    collection  = build_vector_store(faqs, embed_model)

    print(Fore.YELLOW + f"▶ LLM model : {OLLAMA_MODEL}  (make sure Ollama is running)")
    print(Fore.YELLOW + f"▶ Auto-send : {AUTO_SEND}  |  Check every: {CHECK_INTERVAL}s\n")

    while True:
        try:
            print(Fore.CYAN + f"[{datetime.now().strftime('%H:%M:%S')}] Checking inbox...")
            imap = imaplib.IMAP4_SSL(IMAP_SERVER)
            imap.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

            unread = fetch_unread_emails(imap)
            if not unread:
                print("  No new emails.\n")
            else:
                print(Fore.GREEN + f"  Found {len(unread)} unread email(s).\n")

            for em in unread:
                print(Fore.WHITE + f"  ─── Email from: {em['sender']}")
                print(f"      Subject: {em['subject']}")
                print(f"      Body preview: {em['body'][:120]}...")

                # Retrieve relevant FAQ entries
                query = em["subject"] + " " + em["body"]
                hits  = retrieve(query, collection, embed_model)

                print(Fore.YELLOW + "  Top matches:")
                for h in hits:
                    flag = Fore.GREEN if h["score"] >= CONFIDENCE_THRESHOLD else Fore.RED
                    print(f"   {flag}[{h['score']:.2f}] {h['question'][:70]}")

                # Generate reply
                print(Fore.YELLOW + "  Generating reply...")
                reply = generate_reply(em["body"], hits)

                if reply:
                    print(Fore.CYAN + "\n  ── Draft Reply ──────────────────────────")
                    print(textwrap.indent(reply, "  "))
                    print(Fore.CYAN + "  ─────────────────────────────────────────\n")

                    if AUTO_SEND:
                        send_reply(em["sender"], em["subject"], reply)
                        mark_as_read(imap, em["uid"])
                        print(Fore.GREEN + "  ✔ Reply sent automatically.\n")
                    else:
                        confirm = input("  Send this reply? [y/N]: ").strip().lower()
                        if confirm == "y":
                            send_reply(em["sender"], em["subject"], reply)
                            mark_as_read(imap, em["uid"])
                            print(Fore.GREEN + "  ✔ Reply sent.\n")
                        else:
                            print("  ✗ Skipped.\n")
                else:
                    print(Fore.RED + "  ✗ Could not generate reply.\n")

            imap.logout()

        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n\nStopped by user.")
            break
        except Exception as e:
            print(Fore.RED + f"  Error: {e}")

        print(f"  Waiting {CHECK_INTERVAL}s before next check...\n")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
