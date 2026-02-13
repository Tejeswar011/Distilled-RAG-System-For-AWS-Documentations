import weaviate
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import torch

MODEL_PATH = "models/aws_student_model"

# embedding model (for similarity scoring)
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# load student model
student_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
student_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
student_model.eval()

# connect to Weaviate
client = weaviate.connect_to_local()
collection = client.collections.get("AwsDocs")


def retrieve_context(question, top_k=3):
    query_vector = similarity_model.encode(question).tolist()

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


# Test questions
questions = [
    "What is Amazon S3 block public access?",
    "What is IAM used for?",
    "What are EC2 instance types?",
    "What is AWS Lambda?",
    "What is S3 replication?"
]

scores = []

for question in questions:
    context = retrieve_context(question)

    prompt = f"""
Answer the question using only the context.

Context:
{context}

Question:
{question}

Answer:
"""

    teacher_answer = ask_teacher(prompt)
    student_answer = ask_student(prompt)

    # compute similarity
    emb_teacher = similarity_model.encode(teacher_answer, convert_to_tensor=True)
    emb_student = similarity_model.encode(student_answer, convert_to_tensor=True)

    similarity = util.cos_sim(emb_teacher, emb_student).item()
    scores.append(similarity)

    print("\nQuestion:", question)
    print("Relevance score:", round(similarity, 3))

avg_score = sum(scores) / len(scores)
print("\nAverage relevance score:", round(avg_score, 3))

client.close()
