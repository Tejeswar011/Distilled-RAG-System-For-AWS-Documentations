import weaviate
from sentence_transformers import SentenceTransformer
import requests
import json
import random
import os

OUTPUT_FILE = "data/distillation_dataset/dataset.jsonl"

# Load existing questions to avoid duplicates
existing_questions = set()

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                existing_questions.add(data["question"])
            except:
                pass

# embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# connect to Weaviate
client = weaviate.connect_to_local()
collection = client.collections.get("AwsDocs")


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


# get random chunks from database
objects = collection.query.fetch_objects(limit=200).objects
random_chunks = random.sample(objects, min(50, len(objects)))

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:

    for obj in random_chunks:
        context_text = obj.properties["text"]

        # generate questions from context
        question_prompt = f"""
Generate 2 realistic user questions based on the AWS documentation below.

Context:
{context_text}

Questions:
"""

        questions_text = ask_mistral(question_prompt)

        questions = [
            q.strip("- ").strip()
            for q in questions_text.split("\n")
            if q.strip()
        ]

        for question in questions[:2]:

            # skip duplicates
            if question in existing_questions:
                continue

            context = retrieve_context(question)

            answer_prompt = f"""
Answer the question using only the context.

Context:
{context}

Question:
{question}

Answer:
"""

            answer = ask_mistral(answer_prompt)

            record = {
                "question": question,
                "context": context,
                "answer": answer
            }

            f.write(json.dumps(record) + "\n")
            existing_questions.add(question)

            print("Saved:", question)

client.close()
