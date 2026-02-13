import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/ask"

def ask_question(question):
    response = requests.post(
        API_URL,
        json={"question": question}
    )

    if response.status_code == 200:
        return response.json()["answer"]
    else:
        return "Error: Backend not responding"

interface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask an AWS question..."),
    outputs="text",
    title="AWS RAG Assistant (Student Model)",
    description="Ask questions about AWS services. Powered by RAG + Distilled Model."
)

if __name__ == "__main__":
    interface.launch()
