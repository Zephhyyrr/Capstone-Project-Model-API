import os
os.environ["HF_HOME"] = "D:/huggingface" # Set path to Hugging Face cache directory

import gc
import torch
import whisper
import librosa
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from werkzeug.utils import secure_filename

# Konfigurasi
device = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac', 'wma', 'webm', 'amr', 'opus'}

# Init Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load LLaMA 1B pipeline sekali saja
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

    model = whisper.load_model("base").to(device)
    transcription = model.transcribe(audio_path, fp16=False)
    text = transcription.get("text", "").strip()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return jsonify({'text': text})


@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.form
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    # Grammar analysis
    sentences = text.split(". ")
    results = []
    corrected_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not sentence.endswith("."):
            sentence += "."

        # 1. Koreksi Grammar
        correction_prompt = f"""
You are a helpful assistant for grammar correction.

INSTRUCTION:
- Only identify and correct incorrect word(s) or phrase(s) in the sentence.
- Respond in the format: wrong → correct
- Do NOT rewrite the full sentence.
- If the sentence is already correct, respond with: Correct

Sentence: "{sentence}"
Correction:"""

        correction = pipe(correction_prompt, max_new_tokens=50, temperature=0.2, do_sample=False, return_full_text=False)[0]['generated_text'].strip()

        # 2. Alasan Koreksi
        reason_prompt = f"""
INSTRUCTION:
Explain in exactly one sentence **why** the correction above was needed.
- Do NOT rewrite the full sentence.
- Explain correction with bahasa indonesia

Original: "{sentence}"
Correction: "{correction}"
Reason:"""

        reason = pipe(reason_prompt, max_new_tokens=50, temperature=0.3, do_sample=False, return_full_text=False)[0]['generated_text'].strip()

        # 3. Kalimat setelah dikoreksi
        fixed_prompt = f"""
You are a helpful assistant for grammar correction.

INSTRUCTION:
- Correct the grammar of the sentence and return the corrected sentence only.

Sentence: "{sentence}"
Corrected:"""

        corrected = pipe(fixed_prompt, max_new_tokens=100, temperature=0.2, do_sample=False, return_full_text=False)[0]['generated_text'].strip()

        corrected_sentences.append(corrected)
        results.append({
            "kalimat": sentence,
            "koreksi": correction,
            "alasan": reason,
            "setelah_koreksi": corrected
        })

    corrected_paragraph = " ".join(corrected_sentences)

    # Intonasi & Kecepatan Bicara (jika ada audio)
    intonation_analysis = {}
    if 'audio' in request.files:  # Reanalyze only if audio file is included
        file = request.files['audio']
        filename = secure_filename(file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(audio_path)

        y, sr = librosa.load(audio_path, sr=16000)
        pitch, _ = librosa.piptrack(y=y, sr=sr)
        pitch = pitch[pitch > 0] if pitch.size > 0 else np.array([0])
        mean_pitch = float(np.mean(pitch))

        duration = librosa.get_duration(y=y, sr=sr)
        word_count = len(text.split())
        speaking_rate = word_count / duration if duration > 0 else 0

        if speaking_rate < 2.5:
            speed_feedback = "🟡 Terlalu lambat, cobalah berbicara sedikit lebih cepat."
        elif speaking_rate > 4.0:
            speed_feedback = "🟠 Terlalu cepat, cobalah berbicara lebih perlahan."
        else:
            speed_feedback = "🟢 Kecepatan bicara sudah baik."

        intonation_analysis = {
            "mean_pitch": round(mean_pitch, 2),
            "speaking_rate": round(speaking_rate, 2),
            "feedback": speed_feedback
        }

    return jsonify({
        "grammar_analysis": results,
        "corrected_paragraph": corrected_paragraph,
        "intonation_analysis": intonation_analysis
    })


if __name__ == '__main__':
    app.run(debug=True, port=5051)
