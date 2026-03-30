import requests
from bs4 import BeautifulSoup
import csv

url = "https://www.click2refund.com/en/FAQ"
soup = BeautifulSoup(
    requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text, "html.parser"
)

faq_data = []
for h2 in soup.find_all("h2"):
    question = h2.get_text(strip=True)
    answer_parts = []
    for sibling in h2.next_siblings:
        if sibling.name in ("h2", "h4"):
            break
        if sibling.name in ("p", "ul", "ol"):
            answer_parts.append(sibling.get_text(" ", strip=True))
    answer = " ".join(answer_parts).strip()
    if question and answer:
        faq_data.append([question, answer])
