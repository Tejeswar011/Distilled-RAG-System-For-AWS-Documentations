import os
from bs4 import BeautifulSoup

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

for filename in os.listdir(RAW_DIR):
    if not filename.endswith(".html"):
        continue

    path = os.path.join(RAW_DIR, filename)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # remove scripts and styles
    for script in soup(["script", "style", "nav", "footer"]):
        script.extract()

    text = soup.get_text(separator="\n")

    # remove empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    out_path = os.path.join(PROCESSED_DIR, filename.replace(".html", ".txt"))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"Processed {filename}")
