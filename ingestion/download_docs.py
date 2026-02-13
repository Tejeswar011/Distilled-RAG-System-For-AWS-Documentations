import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

SEED_URLS = [
    "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html",
    "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
    "https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html",
    "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html"
]

SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

visited = set()
to_visit = list(SEED_URLS)

MAX_PAGES = 50  # start small

while to_visit and len(visited) < MAX_PAGES:
    url = to_visit.pop(0)

    if url in visited:
        continue

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            continue

        visited.add(url)

        clean_url = url.split("#")[0]  # remove anchors
        filename = clean_url.rstrip("/").split("/")[-1]

        if not filename.endswith(".html"):
            filename = filename + ".html"

        filepath = os.path.join(SAVE_DIR, filename)


        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"Saved {filename}")

        # Extract links
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            full_url = urljoin(url, link["href"])
            if ".pdf" in full_url:
                continue


            if "docs.aws.amazon.com" in full_url:
                if full_url not in visited and full_url not in to_visit:
                    to_visit.append(full_url)

    except Exception as e:
        print("Error:", e)
