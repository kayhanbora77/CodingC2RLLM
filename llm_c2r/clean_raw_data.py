import re
import json


def clean_answer(text):
    # Normalize apostrophes
    text = text.replace("’", "'")

    # Remove excessive whitespace
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)

    # Clean bullet formatting
    text = re.sub(r"\s*-\s*", "\n- ", text)

    # Remove marketing exaggeration phrases (optional)
    text = re.sub(r"Click2Refund", "our legal service", text)

    return text.strip()


def convert_to_instruction_format(input_file, output_file):
    with (
        open(input_file, "r", encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            item = json.loads(line)

            question = item["question"].strip()
            answer = clean_answer(item["answer"])

            formatted = f"""### Instruction:
{question}

### Response:
{answer}

"""

            f_out.write(formatted)


def main():
    input_file = "/home/kayhan/Desktop/LLM/DATASET_C2R/FQA_DATASET.jsonl"
    output_file = "/home/kayhan/Desktop/LLM/DATASET_C2R/cleaned_dataset.txt"
    convert_to_instruction_format(input_file, output_file)


if __name__ == "__main__":
    main()
