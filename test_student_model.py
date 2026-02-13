from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "models/aws_student_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

print("Model loaded successfully")
