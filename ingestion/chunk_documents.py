import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".txt"):
        continue

    path = os.path.join(INPUT_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.strip()

    if not text:
        print(f"Skipped empty file: {filename}")
        continue

    chunks = splitter.split_text(text)

    # if text is short, keep it as one chunk
    if len(chunks) == 0:
        chunks = [text]

    for i, chunk in enumerate(chunks):
        out_file = f"{filename.replace('.txt','')}_chunk_{i}.txt"
        out_path = os.path.join(OUTPUT_DIR, out_file)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(chunk)

    print(f"Chunked {filename} into {len(chunks)} parts")
