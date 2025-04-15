import os
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import librosa
import numpy as np

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Audio file path
audio_file = "audio4.opus"

# Check if the file exists
if not os.path.exists(audio_file):
    raise FileNotFoundError(f"Audio file '{audio_file}' not found!")

# Speech-to-Text (Whisper)
whisper_model = whisper.load_model("base").to(device)
transcription = whisper_model.transcribe(audio_file)
transcribed_text = transcription.get("text", "").strip()

# Grammar Analysis (LLaMA 3.2)
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

prompt = f"Identify the incorrect words in the following sentence and only return the incorrect words:\nSentence: '{transcribed_text}'\nIncorrect words:"
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = pipe(prompt, max_length=100, temperature=0.3)
incorrect_words = response[0]["generated_text"].split("Incorrect words:")[-1].strip()

# Intonation Analysis (Librosa - Pitch & Speaking Rate)
y, sr = librosa.load(audio_file, sr=16000)

# Pitch analysis
pitch, _ = librosa.piptrack(y=y, sr=sr)
pitch = pitch[pitch > 0] if pitch.size > 0 else np.array([0])
mean_pitch = np.mean(pitch)

# Speaking rate analysis
duration = librosa.get_duration(y=y, sr=sr)
word_count = len(transcribed_text.split())
speaking_rate = word_count / duration if duration > 0 else 0

# Evaluasi kecepatan berbicara
if speaking_rate < 2.5:
    speed_feedback = "Terlalu lambat, cobalah berbicara sedikit lebih cepat."
elif speaking_rate > 4.0:
    speed_feedback = "Terlalu cepat, cobalah berbicara lebih perlahan."
else:
    speed_feedback = "Kecepatan berbicara sudah baik."

# Output results
print(f"Transcribed Text: {transcribed_text}")
print(f"Kata yang salah: {incorrect_words}")
print(f"Mean Pitch: {mean_pitch:.2f} Hz")
print(f"Speaking Rate: {speaking_rate:.2f} words/sec")
print(f"Feedback Kecepatan Berbicara: {speed_feedback}")
