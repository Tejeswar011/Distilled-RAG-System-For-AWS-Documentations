import weaviate
from sentence_transformers import SentenceTransformer

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# connect to Weaviate
client = weaviate.connect_to_local()

collection = client.collections.get("AwsDocs")

# test query
query = "How to block public access in S3?"

# convert query to vector
query_vector = model.encode(query).tolist()

# search
results = collection.query.near_vector(
    near_vector=query_vector,
    limit=1
)

# print result
for obj in results.objects:
    print("\nRetrieved Text:\n")
    print(obj.properties["text"][:800])

client.close()
