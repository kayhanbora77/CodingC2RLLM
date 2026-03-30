import sympy.functions.elementary.integers
import sympy.functions.elementary.integers
import sympy.functions.elementary.integers
import importlib
import tiktoken
import re


file_path = "/home/kayhan/Desktop/LLM/DATASET_C2R/cleaned_dataset.txt"


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text):
    # Remove hyphen line breaks like "boarding\n-"
    text = re.sub(r"\n-\n?", "", text)

    # Replace multiple newlines with space
    text = re.sub(r"\n+", " ", text)

    # Lowercase
    text = text.lower()

    return text


def tokenize(text):
    # Keep only words and apostrophes inside words
    tokens = re.findall(r"\b[a-zA-Z’']+\b", text)
    return tokens


def main():
    raw_text = load_text(file_path)

    cleaned = clean_text(raw_text)

    tokenizer = tiktoken.get_encoding("gpt2")
    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)

    strings = tokenizer.decode(integers)
    print(strings)


if __name__ == "__main__":
    main()
