import time
import weaviate
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import torch

MODEL_PATH = "models/aws_student_model"

# load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# load student model
student_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
student_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
student_model.eval()

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


def ask_teacher(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def ask_student(prompt):
    inputs = student_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = student_model.generate(
            **inputs,
            max_new_tokens=150
        )

    text = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("Answer:")[-1].strip()


question = "What is Amazon S3 block public access?"

context = retrieve_context(question)

prompt = f"""
Answer the question using only the context.

Context:
{context}

Question:
{question}

Answer:
"""

# Teacher timing
start = time.time()
teacher_answer = ask_teacher(prompt)
teacher_time = time.time() - start

# Student timing
start = time.time()
student_answer = ask_student(prompt)
student_time = time.time() - start

print("\nTeacher latency:", round(teacher_time, 2), "seconds")
print("Student latency:", round(student_time, 2), "seconds")

print("\nTeacher answer length:", len(teacher_answer.split()), "words")
print("Student answer length:", len(student_answer.split()), "words")


client.close()
