import os
import weaviate
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = "data/chunks"

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# connect to Weaviate
client = weaviate.connect_to_local()

# create collection if not exists
collection_name = "AwsDocs"

if not client.collections.exists(collection_name):
    client.collections.create(
        name=collection_name,
        vectorizer_config=None  # we provide vectors manually
    )

collection = client.collections.get(collection_name)

# load one chunk file
chunk_files = os.listdir(CHUNKS_DIR)
first_file = chunk_files[0]

with open(os.path.join(CHUNKS_DIR, first_file), "r", encoding="utf-8") as f:
    text = f.read()

# generate embedding
vector = model.encode(text).tolist()

# insert into Weaviate
collection.data.insert(
    properties={"text": text},
    vector=vector
)

print("Inserted chunk:", first_file)

client.close()