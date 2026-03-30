"""
Email Auto-Reply System with RAG – draft only, no sending
"""

from httpx import request
import os, csv, time, imaplib, email, textwrap
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from email.utils import parseaddr
from datetime import datetime

import chromadb
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from colorama import Fore, init as colorama_init

# ── Init ────────────────────────────────────────────────────────
load_dotenv()
colorama_init(autoreset=True)

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", 993))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAQ_CSV_PATH_RAW = os.getenv("FAQ_CSV_PATH", "click2refund_faq.csv")
DRAFTS_FOLDER = os.getenv("DRAFTS_FOLDER", "[Gmail]/Drafts")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", 60))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.45))
MARK_AS_SEEN = os.getenv("MARK_AS_SEEN", "true").lower() == "true"

if not os.path.isabs(FAQ_CSV_PATH_RAW):
    FAQ_CSV_PATH = os.path.join(os.path.dirname(__file__), FAQ_CSV_PATH_RAW)
else:
    FAQ_CSV_PATH = FAQ_CSV_PATH_RAW


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
        print(
            Fore.RED
            + f"✘ Loaded 0 FAQ entries from {path}. Check CSV columns and data."
        )
        return []

    print(Fore.GREEN + f"✔ Loaded {len(faqs)} FAQ entries from {path}")
    return faqs


# ── 2. Build ChromaDB vector store ─────────────────────────────
def build_vector_store(faqs: list[dict], embed_model: SentenceTransformer):
    client = chromadb.Client()
    collection = client.get_or_create_collection("faq")

    documents, metadatas, ids = [], [], []
    for i, faq in enumerate(faqs):
        documents.append(faq["question"] + " " + faq["answer"])
        metadatas.append({"question": faq["question"], "answer": faq["answer"]})
        ids.append(str(i))

    embeddings = embed_model.encode(documents, show_progress_bar=True).tolist()
    collection.add(
        documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
    )
    print(Fore.GREEN + f"✔ Vector store ready ({len(documents)} vectors)")
    return collection


# ── 3. Retrieval ────────────────────────────────────────────────
def retrieve(
    query: str, collection, embed_model: SentenceTransformer, top_k: int = 1
) -> list[dict]:
    q_emb = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)

    hits = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        similarity = 1 - dist
        hits.append(
            {
                "question": meta["question"],
                "answer": meta["answer"],
                "score": round(similarity, 3),
            }
        )
    return hits


# ── 4. Generate reply via Ollama ────────────────────────────────
def generate_reply(
    customer_email: str, context_hits: list[dict], max_retries: int = 3
) -> str:
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                print(Fore.YELLOW + f"  Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(2**attempt)
                continue
            print(Fore.RED + f"  LLM Timeout after {max_retries} attempts")
            return None
        except requests.exceptions.RequestException as e:
            
        context_parts = []
        for h in context_hits:
            if h["score"] >= CONFIDENCE_THRESHOLD:
                context_parts.append(f"Q: {h['question']}\nA: {h['answer']}")

    if not context_parts:
        return (
            "Thank you for contacting Click2Refund.\n\n"
            "One of our team members will review your message and get back to you shortly.\n\n"
            "Best regards,\n"
            "Click2Refund Support Team"
        )

    context = "\n\n".join(context_parts)

    prompt = textwrap.dedent(f"""
    You are a helpful customer support agent for Click2Refund, a flight compensation company.
    Use ONLY the FAQ context below to answer the customer's email.
    Be concise, friendly, and professional.
    Do not invent information.
    If the FAQ does not fully answer the issue, politely state that a team member will review it.
    End with:
    Best regards,
    Click2Refund Support Team

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
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(Fore.RED + f"  LLM error: {e}")
        return None


# ── 5. Email utilities ──────────────────────────────────────────
def decode_str(s):
    if not s:
        return ""
    parts = decode_header(s)
    decoded = []
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


def extract_plain_text(msg) -> str:
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()

            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")
                    break
            elif content_type == "text/html" and not body:
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(
                msg.get_content_charset() or "utf-8", errors="replace"
            )

    return body.strip()


def fetch_unread_emails(imap: imaplib.IMAP4_SSL) -> list[dict]:
    imap.select("INBOX")
    status, data = imap.search(None, "UNSEEN")
    if status != "OK":
        return []

    uids = data[0].split()
    emails = []

    for uid in uids:
        status, msg_data = imap.fetch(uid, "(RFC822)")
        if status != "OK" or not msg_data or not msg_data[0]:
            continue

        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)

        subject = decode_str(msg.get("Subject", "(no subject)"))
        sender_raw = decode_str(msg.get("From", ""))
        sender_email = parseaddr(sender_raw)[1]
        body = extract_plain_text(msg)

        emails.append(
            {
                "uid": uid,
                "sender_raw": sender_raw,
                "sender_email": sender_email,
                "subject": subject,
                "body": body,
                "message_id": msg.get("Message-ID", ""),
            }
        )

    return emails


def build_reply_message(to_addr: str, subject: str, body: str) -> MIMEMultipart:
    body = "AUTO-GENERATED DRAFT. HUMAN REVIEW REQUIRED.\n\n" + body

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_addr
    msg["Subject"] = (
        f"Re: {subject}" if not subject.lower().startswith("re:") else subject
    )
    msg.attach(MIMEText(body, "plain", "utf-8"))
    return msg


def save_draft(imap: imaplib.IMAP4_SSL, msg) -> bool:
    try:
        raw_msg = msg.as_bytes()
        result = imap.append(
            DRAFTS_FOLDER, "\\Draft", imaplib.Time2Internaldate(time.time()), raw_msg
        )
        return result[0] == "OK"
    except Exception as e:
        print(Fore.RED + f"  Draft save error: {e}")
        return False


def mark_as_seen(imap: imaplib.IMAP4_SSL, uid):
    imap.store(uid, "+FLAGS", "\\Seen")


# ── 6. Main loop ────────────────────────────────────────────────
def main():
    print(Fore.CYAN + "\n╔══════════════════════════════════════════╗")
    print(Fore.CYAN + "║ Click2Refund Email Draft Generator      ║")
    print(Fore.CYAN + "╚══════════════════════════════════════════╝\n")

    faqs = load_faq(FAQ_CSV_PATH)
    if not faqs:
        print(Fore.RED + "Exiting due to empty FAQ database.")
        return

    print(Fore.YELLOW + f"▶ Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    collection = build_vector_store(faqs, embed_model)

    print(Fore.YELLOW + f"▶ LLM model   : {OLLAMA_MODEL}")
    print(Fore.YELLOW + f"▶ Draft folder: {DRAFTS_FOLDER}")
    print(Fore.YELLOW + f"▶ Check every : {CHECK_INTERVAL}s")
    print(Fore.YELLOW + "▶ Mode        : draft only, no sending\n")

    while True:
        try:
            print(
                Fore.CYAN + f"[{datetime.now().strftime('%H:%M:%S')}] Checking inbox..."
            )

            imap = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
            imap.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

            unread = fetch_unread_emails(imap)
            if not unread:
                print("  No new emails.\n")
            else:
                print(Fore.GREEN + f"  Found {len(unread)} unread email(s).\n")

            for em in unread:
                print(Fore.WHITE + f"  ─── Email from: {em['sender_raw']}")
                print(f"      Subject: {em['subject']}")
                print(f"      Body preview: {em['body'][:120]}...")

                query = f"{em['subject']} {em['body']}"
                hits = retrieve(query, collection, embed_model)

                print(Fore.YELLOW + "  Top matches:")
                for h in hits:
                    flag = (
                        Fore.GREEN if h["score"] >= CONFIDENCE_THRESHOLD else Fore.RED
                    )
                    print(f"   {flag}[{h['score']:.2f}] {h['question'][:70]}")

                print(Fore.YELLOW + "  Generating reply...")
                reply = generate_reply(em["body"], hits)

                if not reply:
                    print(Fore.RED + "  ✗ Could not generate reply.\n")
                    continue

                print(Fore.CYAN + "\n  ── Generated Draft ─────────────────────")
                print(textwrap.indent(reply, "  "))
                print(Fore.CYAN + "  ────────────────────────────────────────")

                if not em["sender_email"]:
                    print(
                        Fore.RED
                        + "  ✗ Could not extract sender email address. Draft not saved.\n"
                    )
                    continue

                draft_msg = build_reply_message(
                    em["sender_email"], em["subject"], reply
                )
                saved = save_draft(imap, draft_msg)

                if saved:
                    print(Fore.GREEN + f"  ✔ Draft saved to '{DRAFTS_FOLDER}'.")
                    if MARK_AS_SEEN:
                        mark_as_seen(imap, em["uid"])
                        print(Fore.GREEN + "  ✔ Original email marked as seen.\n")
                    else:
                        print()
                else:
                    print(Fore.RED + "  ✗ Failed to save draft.\n")

            imap.logout()

        except KeyboardInterrupt:
            print(Fore.YELLOW + "\nStopped by user.")
            break
        except Exception as e:
            print(Fore.RED + f"  Error: {e}")

        print(f"  Waiting {CHECK_INTERVAL}s before next check...\n")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
