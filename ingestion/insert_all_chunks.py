import os
import weaviate
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_DIR = "data/chunks"

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# connect to Weaviate
client = weaviate.connect_to_local()

collection_name = "AwsDocs"

# create collection if not exists
if not client.collections.exists(collection_name):
    client.collections.create(
        name=collection_name,
        vectorizer_config=None
    )

collection = client.collections.get(collection_name)

chunk_files = os.listdir(CHUNKS_DIR)

for filename in tqdm(chunk_files):
    path = os.path.join(CHUNKS_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        continue

    vector = model.encode(text).tolist()

    collection.data.insert(
        properties={"text": text},
        vector=vector
    )

print("All chunks inserted.")

client.close()
