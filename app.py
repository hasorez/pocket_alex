from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
gpt2_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    question = [str(x) for x in request.form.values()][0]
    pred = gpt2_pipeline(question, max_new_tokens=200)
    return render_template("index.html", prediction_text=pred[0]["generated_text"][len(question):].rsplit(".", 1)[0]+ ".")

if __name__ == "__main__":
    app.run()