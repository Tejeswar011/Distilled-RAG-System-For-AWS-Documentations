import weaviate
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# paths
MODEL_PATH = "models/aws_student_model"

# load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# load student model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

model.eval()

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


def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("Answer:")[-1].strip()
    return answer



while True:
    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    context = retrieve_context(question)

    prompt = f"""
Answer the question using only the context.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = generate_answer(prompt)

    print("\nStudent Model Answer:\n")
    print(answer)
