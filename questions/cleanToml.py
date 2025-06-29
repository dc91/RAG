import os
# import re
from glob import glob
from tomlkit import parse, dumps

# Directories
input_dir = "questions/raw"
output_dir = "questions/cleaned/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Helper functions
def clean_answer(answer):
    # answer = re.sub(r' \[cite:\s*\d+\]', '', answer)
    # answer = answer.rstrip()
    if answer.endswith('.'):
        answer = answer[:-1]
    return answer

def capitalize_category(text):
    return text[:1].upper() + text[1:] if text else text

# Process all .toml files in the input directory
for file_path in glob(os.path.join(input_dir, "*.toml")):
    with open(file_path, "r", encoding="utf-8") as f:
        doc = parse(f.read())

    # Clean each question item
    for item in doc.get("questions", []):
        if "answer" in item:
            item["answer"] = clean_answer(item["answer"])
        if "category" in item:
            item["category"] = capitalize_category(item["category"])
        if "difficulty" in item:
            item["difficulty"] = capitalize_category(item["difficulty"])

    # Prepare output file path
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"cleaned_{file_name}")

    # Write cleaned data
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dumps(doc))
