import os
import gc
import torch
import whisper
import librosa
import numpy as np
import nltk
import json
import re
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from werkzeug.utils import secure_filename

# Unduh tokenizer NLTK
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Konfigurasi dasar
device = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_FOLDER = "uploads"
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac', 'wma', 'webm', 'amr', 'opus'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load LLaMA model sekali saja
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'Audio file is required'}), 400
    file = request.files['audio']
    filename = secure_filename(file.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(audio_path)

    model_whisper = whisper.load_model("base").to(device)
    transcription = model_whisper.transcribe(audio_path, fp16=False)
    text = transcription.get("text", "").strip()

    del model_whisper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return jsonify({'text': text})

def parse_model_response(response_text):
    # Coba JSON parse langsung
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback pakai regex sederhana
        match = re.search(r'\{.*?"original":.*?"corrected":.*?"reason":.*?\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None
    return None

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.form
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    sentences = sent_tokenize(text)
    results = []
    corrected_sentences = []

    for sentence in sentences:
        structured_prompt = f"""
You are a grammar correction assistant. For the given sentence, return a JSON with:
- "original": original sentence
- "corrected": corrected sentence
- "reason": short explanation of the correction

Example:
- "original": "Yesterday I go to the store."
- "corrected": "Yesterday I went to the store.",
- "reason": "Incorrect verb tense because 'go' should be 'went' in past tense."

Sentence: "{sentence}"

Respond only in valid JSON format.
""".strip()

        response = pipe(structured_prompt, max_new_tokens=120, temperature=0.2, do_sample=False, return_full_text=False)[0]['generated_text']
        parsed = parse_model_response(response)

        if parsed:
            original = parsed.get("original", sentence)
            corrected = parsed.get("corrected", sentence)
            reason = parsed.get("reason", "No reason provided")
        else:
            original = sentence
            corrected = sentence
            reason = "Failed to parse model output."

        corrected_sentences.append(corrected)
        results.append({
            "original": original,
            "corrected": corrected,
            "reason": reason
        })

    corrected_paragraph = " ".join(corrected_sentences)

    return jsonify({
        "success": True,
        "message": "Analysis successful",
        "data": {
            "corrected_paragraph": corrected_paragraph,
            "grammar_analysis": results
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
