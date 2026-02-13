from fastapi import FastAPI
from pydantic import BaseModel
from backend.inference import generate_answer


app = FastAPI(title="AWS RAG Student Model API")


class Query(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"message": "RAG API is running"}


@app.post("/ask")
def ask_question(query: Query):
    answer = generate_answer(query.question)
    return {"answer": answer}
