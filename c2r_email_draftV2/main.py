"""
Email Auto-Reply System with RAG for Click2Refund
Optimized for Python 3.13 - Using OLLAMA for Embeddings (No PyTorch needed)
"""

import os
import csv
import time
import imaplib
import email
import textwrap
import json
import hashlib
import logging
import pickle
import math
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from email.utils import parseaddr
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import requests
from dotenv import load_dotenv
from colorama import init as colorama_init, Fore, Style

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
colorama_init(autoreset=True)


@dataclass
class Config:
    # Email settings
    email_address: str = os.getenv("EMAIL_ADDRESS", "")
    email_password: str = os.getenv("EMAIL_PASSWORD", "")
    imap_server: str = os.getenv("IMAP_SERVER", "imap.gmail.com")
    imap_port: int = int(os.getenv("IMAP_PORT", 993))

    # RAG settings
    # We use nomic-embed-text for embeddings, phi3 for chat
    ollama_llm_model: str = os.getenv("OLLAMA_MODEL", "phi3:mini")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    faq_csv_path: str = os.getenv("FAQ_CSV_PATH", "click2refund_faq.csv")
    # We save embeddings as a simple numpy file
    embeddings_path: str = os.getenv("EMBEDDINGS_PATH", "faq_embeddings.npy")
    metadata_path: str = os.getenv("METADATA_PATH", "faq_metadata.pkl")

    # Behavior settings
    check_interval: int = int(os.getenv("CHECK_INTERVAL_SECONDS", 90))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.40))
    mark_as_seen: bool = os.getenv("MARK_AS_SEEN", "true").lower() == "true"
    drafts_folder: str = os.getenv("DRAFTS_FOLDER", "[Gmail]/Drafts")

    # Advanced settings
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", 180))
    processed_log_path: str = os.getenv("PROCESSED_LOG_PATH", "processed_emails.json")
    low_confidence_log_path: str = os.getenv(
        "LOW_CONFIDENCE_LOG_PATH", "low_confidence_log.json"
    )

    def validate(self) -> bool:
        if not self.email_address or not self.email_password:
            logging.error("EMAIL_ADDRESS and EMAIL_PASSWORD must be set")
            return False
        if not os.path.exists(self.faq_csv_path):
            logging.error(f"FAQ CSV file not found: {self.faq_csv_path}")
            return False
        return True


config = Config()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("email_auto_reply.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# --- NEW: Ollama Embedder Class (Replaces SentenceTransformers) ---
class OllamaEmbedder:
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api/embed"
        logger.info(f"Initialized Ollama Embedder with model: {model_name}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from Ollama for a list of texts."""
        embeddings = []

        logger.info(f"Generating embeddings for {len(texts)} items via Ollama...")
        for i, text in enumerate(texts):
            try:
                response = requests.post(
                    self.base_url,
                    json={"model": self.model_name, "prompt": text},
                    timeout=60,
                )
                response.raise_for_status()
                embedding = np.array(response.json().get("embedding", []))
                embeddings.append(embedding)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(texts)}...")
            except Exception as e:
                logger.error(f"Error embedding text {i}: {e}")
                # Fallback: zero vector
                embeddings.append(np.zeros(768))

        return np.array(embeddings).astype("float32")


# --- Helper Trackers (Same as before) ---
class ProcessedEmailTracker:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.processed = set()
        self._load()

    def _load(self):
        if self.log_path.exists():
            try:
                with open(self.log_path, "r") as f:
                    data = json.load(f)
                    self.processed = set(data.get("processed_ids", []))
                logger.info(f"Loaded {len(self.processed)} processed email IDs")
            except Exception as e:
                logger.warning(f"Could not load processed log: {e}")

    def save(self):
        try:
            with open(self.log_path, "w") as f:
                json.dump({"processed_ids": list(self.processed)}, f)
        except Exception as e:
            logger.error(f"Could not save processed log: {e}")

    def is_processed(self, email_id: str) -> bool:
        return email_id in self.processed

    def mark_processed(self, email_id: str):
        self.processed.add(email_id)
        if len(self.processed) % 10 == 0:
            self.save()

    def save_final(self):
        self.save()


class LowConfidenceTracker:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)

    def log(self, email_data: Dict, top_score: float, retrieved_questions: List[str]):
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "from": email_data.get("sender_email", "unknown"),
                "subject": email_data.get("subject", ""),
                "body_preview": email_data.get("body", "")[:500],
                "top_score": top_score,
                "retrieved_questions": retrieved_questions[:3],
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(
                f"Logged low-confidence email from {email_data.get('sender_email')}"
            )
        except Exception as e:
            logger.error(f"Could not log low confidence email: {e}")


def load_faq(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        logger.error(f"FAQ file not found: {path}")
        return []

    # We will try to load with different delimiters
    delimiters_to_try = [",", "\t"]

    for delimiter in delimiters_to_try:
        try:
            # Open file fresh for each attempt
            with open(path, encoding="utf-8", newline="") as f:
                # Peek at the first line to see if it makes sense with this delimiter
                first_line = f.readline()
                f.seek(0)  # Reset to start

                # Simple heuristic: if the delimiter isn't in the header, skip it (unless it's the last resort)
                # But we will try both anyway to be safe against mismatched headers/data

                reader = csv.DictReader(f, delimiter=delimiter)

                # Clean headers (handle potential BOM or spaces)
                reader.fieldnames = [name.strip() for name in reader.fieldnames]

                # Check if we have the required columns
                if (
                    "Question" not in reader.fieldnames
                    or "Answer" not in reader.fieldnames
                ):
                    # Try to find case-insensitive matches just in case
                    found_q = any(
                        "question" in name.lower() for name in reader.fieldnames
                    )
                    found_a = any(
                        "answer" in name.lower() for name in reader.fieldnames
                    )
                    if not (found_q and found_a):
                        continue  # Skip this delimiter, try the next one

                faqs = []
                for row in reader:
                    # Get values using exact keys found
                    q_key = next(
                        (k for k in reader.fieldnames if "question" in k.lower()),
                        "Question",
                    )
                    a_key = next(
                        (k for k in reader.fieldnames if "answer" in k.lower()),
                        "Answer",
                    )

                    q = row.get(q_key, "").strip()
                    a = row.get(a_key, "").strip()

                    if q and a:
                        faqs.append({"question": q, "answer": a})

                if faqs:
                    logger.info(
                        f"Successfully loaded {len(faqs)} FAQs using delimiter: '{delimiter}'"
                    )
                    return faqs

        except Exception as e:
            logger.debug(f"Failed to load with delimiter '{delimiter}': {e}")
            continue

    logger.error("Could not load FAQ entries with either Comma or Tab delimiters.")
    logger.error("Please check your CSV file format.")
    return []


# --- NEW: Numpy Vector Store (Replaces FAISS) ---
class SimpleVectorStore:
    def __init__(
        self,
        faqs: List[Dict],
        embedder: OllamaEmbedder,
        index_path: str,
        metadata_path: str,
    ):
        self.faqs = faqs
        self.embedder = embedder
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embeddings = None
        self.metadata = []
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.embeddings = np.load(self.index_path)
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(
                    f"Loaded existing embeddings with {len(self.embeddings)} vectors"
                )
                return
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}. Recreating...")

        self._create_index()

    def _create_index(self):
        logger.info("Creating new embeddings via Ollama (this might take a moment)...")
        documents = [f"{faq['question']} {faq['answer']}" for faq in self.faqs]

        # Call Ollama to get vectors
        self.embeddings = self.embedder.encode(documents)

        self.metadata = [
            {"question": faq["question"], "answer": faq["answer"]} for faq in self.faqs
        ]

        # Save to disk
        np.save(self.index_path, self.embeddings)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Embeddings created and saved.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.embeddings is None:
            return []

        try:
            # 1. Embed the query
            query_vector = self.embedder.encode([query])[0]

            # 2. Calculate Cosine Similarity manually using Numpy
            # Formula: (A . B) / (||A|| * ||B||)
            norms = np.linalg.norm(self.embeddings, axis=1)
            query_norm = np.linalg.norm(query_vector)

            # Dot product
            dots = np.dot(self.embeddings, query_vector)

            # Similarity scores
            similarities = dots / (norms * query_norm + 1e-8)

            # 3. Get top_k indices
            # argsort sorts ascending, so we take the last k and reverse them
            top_indices = similarities.argsort()[-top_k:][::-1]

            hits = []
            for idx in top_indices:
                hits.append(
                    {
                        "question": self.metadata[idx]["question"],
                        "answer": self.metadata[idx]["answer"],
                        "score": float(similarities[idx]),
                    }
                )
            return hits
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []


# --- Reply Generator (Same logic, just updated class usage) ---
class ReplyGenerator:
    def __init__(self, model: str, timeout: int = 180, max_retries: int = 3):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = "http://localhost:11434/api/generate"

    def generate(
        self,
        customer_email: Dict,
        context_hits: List[Dict],
        confidence_threshold: float,
    ) -> Optional[str]:
        relevant_context = [
            h for h in context_hits if h["score"] >= confidence_threshold
        ]

        if not relevant_context:
            logger.info("No relevant FAQ found, returning generic reply")
            return self._get_generic_reply()

        context_parts = [
            f"Q: {h['question']}\nA: {h['answer']}" for h in relevant_context
        ]
        context = "\n\n".join(context_parts)

        prompt = textwrap.dedent(f"""
        You are a helpful customer support agent for Click2Refund.
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
        Subject: {customer_email.get("subject", "No subject")}
        Body: {customer_email.get("body", "")[:1000]}

        --- YOUR REPLY ---
        """).strip()

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                reply = response.json().get("response", "").strip()
                if reply:
                    logger.info(f"Generated reply ({len(reply)} chars)")
                    return reply
            except requests.exceptions.Timeout:
                logger.warning(
                    f"Ollama timeout (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                break

        return self._get_generic_reply()

    def _get_generic_reply(self) -> str:
        return textwrap.dedent("""
        Thank you for contacting Click2Refund.
        We have received your inquiry and one of our team members will review your case and get back to you shortly.
        Best regards,
        Click2Refund Support Team
        """).strip()


# --- Email Processor (Mostly unchanged) ---
class EmailProcessor:
    def __init__(
        self, config: Config, vector_store: SimpleVectorStore, reply_gen: ReplyGenerator
    ):
        self.config = config
        self.vector_store = vector_store
        self.reply_gen = reply_gen
        self.processed_tracker = ProcessedEmailTracker(config.processed_log_path)
        self.low_confidence_tracker = LowConfidenceTracker(
            config.low_confidence_log_path
        )

    def connect_imap(self) -> imaplib.IMAP4_SSL:
        imap = imaplib.IMAP4_SSL(self.config.imap_server, self.config.imap_port)
        imap.login(self.config.email_address, self.config.email_password)
        return imap

    def fetch_unread_emails(self, imap: imaplib.IMAP4_SSL) -> List[Dict]:
        imap.select("INBOX")
        status, data = imap.search(None, "UNSEEN")
        if status != "OK":
            return []

        uids = data[0].split()
        emails = []

        for uid in uids:
            try:
                status, msg_data = imap.fetch(uid, "(RFC822)")
                if status != "OK" or not msg_data:
                    continue

                msg = email.message_from_bytes(msg_data[0][1])
                subject = self._decode_header(msg.get("Subject", ""))
                sender_raw = self._decode_header(msg.get("From", ""))
                sender_email = parseaddr(sender_raw)[1]
                body = self._extract_plain_text(msg)

                unique_id = hashlib.md5(
                    f"{msg.get('Message-ID', '')}_{sender_email}".encode()
                ).hexdigest()

                emails.append(
                    {
                        "uid": uid,
                        "unique_id": unique_id,
                        "sender_email": sender_email,
                        "subject": subject,
                        "body": body,
                    }
                )
            except Exception as e:
                logger.error(f"Error fetching email: {e}")
                continue

        return emails

    def _decode_header(self, header: str) -> str:
        if not header:
            return ""
        parts = decode_header(header)
        decoded = []
        for part, enc in parts:
            if isinstance(part, bytes):
                decoded.append(part.decode(enc or "utf-8", errors="replace"))
            else:
                decoded.append(part)
        return "".join(decoded)

    def _extract_plain_text(self, msg) -> str:
        body = ""
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain" and not part.get(
                        "Content-Disposition"
                    ):
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or "utf-8"
                            body = payload.decode(charset, errors="replace")
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
        return body.strip()

    def save_draft(
        self, imap: imaplib.IMAP4_SSL, to_addr: str, subject: str, body: str
    ) -> bool:
        try:
            full_body = "AUTO-GENERATED DRAFT - HUMAN REVIEW REQUIRED\n\n" + body
            msg = MIMEMultipart()
            msg["From"] = self.config.email_address
            msg["To"] = to_addr
            msg["Subject"] = (
                f"Re: {subject}" if not subject.lower().startswith("re:") else subject
            )
            msg.attach(MIMEText(full_body, "plain", "utf-8"))

            # Try to save
            result = imap.append(
                self.config.drafts_folder,
                "\\Draft",
                imaplib.Time2Internaldate(time.time()),
                msg.as_bytes(),
            )

            # CHECK FOR IMAP ERRORS (The previous code missed this!)
            status, data = result
            if status != "OK":
                logger.error(f"IMAP Error saving draft: {status} - {data}")
                logger.error(f"Folder attempted: {self.config.drafts_folder}")
                return False

            return True
        except Exception as e:
            logger.error(f"Exception saving draft: {e}")
            return False

    def mark_as_seen(self, imap: imaplib.IMAP4_SSL, uid):
        if self.config.mark_as_seen:
            try:
                imap.store(uid, "+FLAGS", "\\Seen")
            except Exception as e:
                logger.error(f"Failed to mark as seen: {e}")

    def process_email(self, imap: imaplib.IMAP4_SSL, email_data: Dict) -> bool:
        try:
            logger.info(
                f"Processing: {email_data['sender_email']} - {email_data['subject'][:50]}"
            )

            query = f"{email_data['subject']} {email_data['body'][:500]}"
            hits = self.vector_store.retrieve(query, top_k=3)

            if hits:
                logger.info(
                    f"Top match: {hits[0]['score']:.3f} - {hits[0]['question'][:50]}"
                )
                if hits[0]["score"] < self.config.confidence_threshold:
                    retrieved_questions = [h["question"] for h in hits]
                    self.low_confidence_tracker.log(
                        email_data, hits[0]["score"], retrieved_questions
                    )

            reply = self.reply_gen.generate(
                email_data, hits, self.config.confidence_threshold
            )
            if not reply:
                return False

            success = self.save_draft(
                imap, email_data["sender_email"], email_data["subject"], reply
            )

            reply = self.reply_gen.generate(
                email_data, hits, self.config.confidence_threshold
            )
            if not reply:
                return False

            # NEW: Only save draft if it's NOT the generic "we will review" reply
            # This prevents filling your drafts with spam replies.
            if "team member will review your case" in reply:
                logger.info("Skipping draft save for generic/unrelated email.")
                return True  # Count as processed, but no draft saved

            success = self.save_draft(
                imap, email_data["sender_email"], email_data["subject"], reply
            )

            if success:
                self.mark_as_seen(imap, email_data["uid"])
                return True
            return False

        except Exception as e:
            logger.error(f"Error processing email: {e}", exc_info=True)
            return False


# --- Main App ---
class EmailAutoReplyApp:
    def __init__(self):
        self.config = config
        if not self.config.validate():
            raise ValueError("Invalid configuration")

        logger.info("=" * 60)
        logger.info("Click2Refund Email Auto-Reply (Python 3.13 + Ollama)")
        logger.info(f"LLM Model: {self.config.ollama_llm_model}")
        logger.info(f"Embed Model: {self.config.ollama_embed_model}")
        logger.info("=" * 60)

        # 1. Initialize Ollama Embedder (No PyTorch!)
        logger.info("Initializing Ollama Embedder...")
        self.embedder = OllamaEmbedder(self.config.ollama_embed_model)

        # 2. Load FAQ
        logger.info("Loading FAQ database...")
        self.faqs = load_faq(self.config.faq_csv_path)
        if not self.faqs:
            raise ValueError("No FAQ entries loaded")

        # 3. Build Vector Store (Using Numpy + Ollama)
        logger.info("Building vector store (this sends data to Ollama)...")
        self.vector_store = SimpleVectorStore(
            self.faqs,
            self.embedder,
            self.config.embeddings_path,
            self.config.metadata_path,
        )

        # 4. Initialize Reply Generator
        logger.info("Initializing reply generator...")
        self.reply_gen = ReplyGenerator(self.config.ollama_llm_model)

        self.email_processor = EmailProcessor(
            self.config, self.vector_store, self.reply_gen
        )
        self.stats = {
            "emails_processed": 0,
            "drafts_saved": 0,
            "errors": 0,
            "start_time": datetime.now(),
        }

    def print_status(self):
        runtime = datetime.now() - self.stats["start_time"]
        print(Fore.CYAN + "\n" + "=" * 60)
        print(
            Fore.CYAN
            + f"📊 Stats - Runtime: {runtime.seconds // 60}m {runtime.seconds % 60}s"
        )
        print(
            Fore.CYAN
            + f"   Emails: {self.stats['emails_processed']} | Drafts: {self.stats['drafts_saved']} | Errors: {self.stats['errors']}"
        )
        print(Fore.CYAN + "=" * 60 + Style.RESET_ALL)

    def run(self):
        logger.info("Starting main loop...")
        while True:
            try:
                imap = self.email_processor.connect_imap()
                unread = self.email_processor.fetch_unread_emails(imap)

                if unread:
                    logger.info(f"Found {len(unread)} unread email(s)")
                    for email_data in unread:
                        if self.email_processor.processed_tracker.is_processed(
                            email_data["unique_id"]
                        ):
                            continue

                        success = self.email_processor.process_email(imap, email_data)
                        self.stats["emails_processed"] += 1
                        if success:
                            self.stats["drafts_saved"] += 1
                            self.email_processor.processed_tracker.mark_processed(
                                email_data["unique_id"]
                            )
                        else:
                            self.stats["errors"] += 1

                        if self.stats["emails_processed"] % 5 == 0:
                            self.print_status()

                imap.logout()
                self.email_processor.processed_tracker.save_final()

                if self.stats["emails_processed"] > 0:
                    self.print_status()

                logger.info(f"Waiting {self.config.check_interval} seconds...\n")
                time.sleep(self.config.check_interval)

            except KeyboardInterrupt:
                logger.info("\nShutting down gracefully...")
                self.email_processor.processed_tracker.save_final()
                self.print_status()
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                self.stats["errors"] += 1
                time.sleep(30)


def main():
    try:
        app = EmailAutoReplyApp()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start: {e}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
