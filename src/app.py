import os
import gc
import torch
import whisper
import nltk
import json
import re
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from werkzeug.utils import secure_filename

# Unduh resource NLTK
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Konfigurasi
os.environ["HF_HOME"] = "D:/huggingface"
device = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_FOLDER = "uploads"
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac', 'wma', 'webm', 'amr', 'opus'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load LLaMA model
model_path = r"D:/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct\snapshots\9213176726f574b556790deb65791e0c5aa438b6"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Fungsi pengecekan file audio
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

# Fungsi parsing output model
import re

def parse_model_response(response_text):
    pattern = re.compile(
        r'Original:\s*(.*?)\s*Corrected:\s*(.*?)\s*Reason:\s*(.*)', 
        re.DOTALL
    )

    match = pattern.search(response_text)
    if match:
        original = match.group(1).strip()
        corrected = match.group(2).strip()
        reason = match.group(3).strip()

        # Safety check
        if not corrected or not original:
            return None

        return {
            "original": original,
            "corrected": corrected,
            "reason": reason
        }

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
You are an English grammar assistant. Correct the grammar and explain why.

Sentence: {sentence}

Format your response exactly like this:
Original: <original sentence>
Corrected: <corrected sentence>
Reason: <short explanation>
""".strip()

        response = pipe(
            structured_prompt,
            max_new_tokens=120,
            temperature=0.2,
            do_sample=False,
            return_full_text=False
        )[0]['generated_text']

        # Debug log
        print(f"Model response:\n{response}\n")

        parsed = parse_model_response(response)

        if parsed:
            original = parsed["original"]
            corrected = parsed["corrected"]
            reason = parsed["reason"]
        else:
            original = sentence
            corrected = sentence
            reason = f"Failed to parse model output. Model said: {response}"

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
    app.run(debug=True, port=5051)
