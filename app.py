from flask import Flask, request, render_template
from transformers import MarianMTModel, MarianTokenizer
import torch

# Load the translation model (English -> French)
model_name = 'Helsinki-NLP/opus-mt-en-fr'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MarianMTModel.from_pretrained(model_name).to(device)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate_text(text):
    if not text.strip():
        return "Please enter text to translate."
    
    with torch.no_grad():
        translated = tokenizer.encode(text, return_tensors="pt", truncation=True, padding=True).to(device)
        translated_output = model.generate(translated, num_beams=5, max_length=100, early_stopping=True)
        translated_text = tokenizer.decode(translated_output[0], skip_special_tokens=True)
    
    return translated_text

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    translated_text = ""
    if request.method == "POST":
        input_text = request.form["input_text"]
        translated_text = translate_text(input_text)
    return render_template("index.html", translated_text=translated_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
