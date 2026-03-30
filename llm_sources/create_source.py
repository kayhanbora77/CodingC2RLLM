import pyodbc
from bs4 import BeautifulSoup
from pathlib import Path

# ===============================
# CONFIG
# ===============================
CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=C2RBetaDB;"
    "UID=SA;"
    "PWD=12.September.2025;"
    "Encrypt=no;"
)

OUTPUT_FILE = Path("ComplaintEmails.txt")


# ===============================
# HTML CLEAN FUNCTION
# ===============================
def clean_html(html_text: str) -> str:
    if not html_text:
        return ""

    soup = BeautifulSoup(html_text, "html.parser")

    # Remove script & style
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Get clean text
    text = soup.get_text(separator="\n")

    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


# ===============================
# MAIN
# ===============================
def main():
    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()

    query = """
    SELECT Subject, Body
    FROM ComplaintEmails
    ORDER BY ReceiveDate
    """
    cursor.execute(query)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, row in enumerate(cursor.fetchall(), start=1):
            subject = row.Subject or ""
            body_html = row.Body or ""

            body_text = clean_html(body_html)

            f.write(f"================ EMAIL {idx} ================\n")
            f.write(f"Subject:\n{subject}\n\n")
            f.write("Body:\n")
            f.write(body_text)
            f.write("\n\n")

    cursor.close()
    conn.close()

    print("✅ TXT file created:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
