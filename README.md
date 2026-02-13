# Distilled RAG System for AWS Documentation

An end-to-end Retrieval-Augmented Generation (RAG) system that answers AWS documentation queries using a distilled language model.  
The project demonstrates model distillation, semantic retrieval, API deployment, UI integration, and monitoring.
This project demonstrates an end-to-end applied ML pipeline including retrieval, model distillation, API deployment, UI integration, and monitoring.

---

## Project Overview

This project builds a complete AI application pipeline:

1. AWS documentation is downloaded and processed
2. Documents are chunked and stored in a vector database
3. A teacher LLM generates a dataset
4. A smaller student model is trained via distillation
5. The student model serves answers through an API
6. A frontend UI allows interaction
7. MLflow monitors performance metrics

---

## Key Achievements

- Reduced inference latency by ~60%
- Reduced model size by ~54%
- Maintained ~0.90 semantic relevance score
- Built a full-stack ML system with monitoring

---

## Tech Stack

- Python
- FastAPI
- Gradio
- Transformers (HuggingFace)
- Sentence Transformers
- Weaviate
- MLflow
- Docker

---

## Project Structure

## Project Structure

```bash
model_distillation/
│
├── backend/        # FastAPI backend and inference
├── frontend/       # Gradio UI
├── config/         # YAML configuration files
├── ingestion/      # Data ingestion pipeline
├── evaluation/     # Benchmark scripts
├── rag_teacher/    # Dataset generation and distillation
├── models/         # Saved student model
├── data/           # Raw and processed documents
```


---

## System Architecture

<img width="482" height="541" alt="system_architechture" src="https://github.com/user-attachments/assets/74e8d1cd-0ed2-465e-a370-7b806fd251d4" />


---

## How to Run the Project

### Step 1 — Clone Repository
```bash
git clone https://github.com/Tejeswar011/Distilled-RAG-System-For-AWS-Documentations
cd model_distillation
```
---

### Step 2 — Create Environment

Using uv:
```bash
uv venv
.venv\Scripts\activate
```

Install dependencies:
```bash
uv pip install -r requirements.txt
```
---

### Step 3- Start Teacher Model (ollama)
```bash
ollama run mistral
```

---

### Step 4 — Start Vector Database (Weaviate)

Run Docker container:
```bash
docker run -d -p 8080:8080 semitechnologies/weaviate:latest
```

verify:
```bash
http://localhost:8080/v1/meta
```

---

### Step 5 — Start MLflow
```bash
mlflow ui
```

Open:
```bash
http://127.0.0.1:5000
```

---

### Step 6 — Start Backend
```bash
uvicorn backend.app:app --reload
```

Backend will run at:
```bash
http://127.0.0.1:8000/docs
```

---

### Step 7 — Start Frontend

Open a new terminal:
```bash
python frontend/app.py
```

Frontend will run at:
```bash
http://127.0.0.1:7860
```

---

## Example Questions

- What is Amazon S3 block public access?
- What are EC2 instance types?
- What is AWS Lambda used for?

---

## Evaluation Metrics

The system measures:

- Latency
- Model size
- Semantic relevance score
- Answer length

Metrics are logged and visualized in MLflow.

---

## Configuration

Configuration files are stored in:


These allow tuning:
- retrieval settings
- generation parameters
- model paths

without modifying code.

---

## Future Improvements

- GPU inference optimization
- Quantization for faster response
- FAISS deployment version
- Authentication layer for API
- Kubernetes deployment

---

## Author

N Tejeswar Reddy 

B.Tech Final Year  
AI/ML Enthusiast  

---

## License

MIT License
