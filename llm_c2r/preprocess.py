import json
import re

file_path = "/home/kayhan/Desktop/LLM/DATASET_C2R/cleaned_dataset.txt"

try:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        raw_data = f.read()
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# Normalize spacing around markers (helps a lot)
raw_data = re.sub(r"\s*###\s*Instruction\s*:\s*", "### Instruction: ", raw_data)
raw_data = re.sub(r"\s*###\s*Response\s*:\s*", "### Response: ", raw_data)

# Better regex: allows flexible whitespace, non-greedy capture
pattern = r"### Instruction: (.*?)\s*### Response: (.*?)(?=### Instruction:|\Z)"
pairs = re.findall(pattern, raw_data, re.DOTALL | re.MULTILINE)

print(f"Found {len(pairs)} instruction-response pairs\n")

dataset = []

for i, (instruction, response) in enumerate(pairs, 1):
    instruction = instruction.strip()
    response = response.strip().replace("our legal service", "Click2Refund")
    # Optional: fix bullet points
    response = re.sub(r"\s*-\s*", "\n- ", response)

    entry = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful flight compensation expert at Click2Refund. Answer factually, politely, and mention our no-win-no-fee service when relevant.",
            },
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }
    dataset.append(entry)

    # Show first few for debugging
    if i <= 3:
        print(f"Pair {i}:")
        print(
            "Q:", instruction[:100] + "..." if len(instruction) > 100 else instruction
        )
        print("A:", response[:100] + "..." if len(response) > 100 else response)
        print("-" * 60)

# Save
output_file = "/home/kayhan/Desktop/LLM/DATASET_C2R/faq_dataset.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\nSaved {len(dataset)} examples to {output_file}")
