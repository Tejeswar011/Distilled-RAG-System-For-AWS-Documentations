import weaviate
from sentence_transformers import SentenceTransformer
import requests

# embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# connect to Weaviate
client = weaviate.connect_to_local()
collection = client.collections.get("AwsDocs")

def retrieve_context(question, top_k=3):
    query_vector = embed_model.encode(question).tolist()

    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k
    )

    context = "\n\n".join(
        obj.properties["text"] for obj in results.objects
    )

    return context

def ask_mistral(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# main loop
while True:
    question = input("\nAsk a question (or type exit): ")
    if question.lower() == "exit":
        break

    context = retrieve_context(question)

    prompt = f"""
You are an AWS assistant. Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = ask_mistral(prompt)

    print("\nAnswer:\n")
    print(answer)

client.close()
