import weaviate
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import mlflow
import time
import yaml

with open("configs/rag_config.yaml") as f:
    rag_config = yaml.safe_load(f)

TOP_K = rag_config["retrieval"]["top_k"]
MAX_TOKENS = rag_config["generation"]["max_new_tokens"]


MODEL_PATH = "models/aws_student_model"

# load once (important)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

client = weaviate.connect_to_local()
collection = client.collections.get("AwsDocs")


def retrieve_context(question, top_k=TOP_K):
    query_vector = embed_model.encode(question).tolist()

    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k
    )

    context = "\n\n".join(
        obj.properties["text"] for obj in results.objects
    )

    return context


def generate_answer(question):

    start_time = time.time()

    context = retrieve_context(question)

    prompt = f"""
Answer the question using only the context.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = text.split("Answer:")[-1].strip()

    latency = time.time() - start_time

    # MLflow logging
    with mlflow.start_run(nested=True):
        mlflow.log_param("question", question[:100])
        mlflow.log_metric("latency", latency)
        mlflow.log_metric("answer_length", len(answer.split()))

    return answer

