import weaviate
from sentence_transformers import SentenceTransformer

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# connect to Weaviate
client = weaviate.connect_to_local()
collection = client.collections.get("AwsDocs")

query = input("Enter your question: ")

# convert query to vector
query_vector = model.encode(query).tolist()

# retrieve top 3 chunks
results = collection.query.near_vector(
    near_vector=query_vector,
    limit=3
)

print("\nRetrieved Context:\n")

for i, obj in enumerate(results.objects):
    print(f"\n--- Chunk {i+1} ---\n")
    print(obj.properties["text"][:800])

client.close()
