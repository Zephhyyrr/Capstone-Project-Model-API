# Import library
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

whisper_model = whisper.load_model("base").to(device) 
transcription = whisper_model.transcribe("audio4.opus")

transcribed_text = transcription["text"]
print("Transcribed Text:", transcribed_text)

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

prompt = f"""
Please check the following sentence for grammatical errors and correct it:

Sentence: "{transcribed_text}"

Corrected Sentence:
"""

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = pipe(prompt, max_length=512, temperature=0.7)

print("Grammar Analysis and Correction:", response[0]['generated_text'])
