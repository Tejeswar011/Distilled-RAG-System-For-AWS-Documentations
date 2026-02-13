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


---

## System Architecture


---

## How to Run the Project

### Step 1 — Clone Repository


---

### Step 2 — Create Environment

Using uv:


Install dependencies:


---

### Step 3 — Start Vector Database (Weaviate)

Run Docker container:


---

### Step 4 — Start MLflow


Open:


---

### Step 5 — Start Backend


Backend will run at:


---

### Step 6 — Start Frontend

Open a new terminal:


Frontend will run at:


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

Your Name  
B.Tech Final Year  
AI/ML Enthusiast  

---

## License

MIT License
