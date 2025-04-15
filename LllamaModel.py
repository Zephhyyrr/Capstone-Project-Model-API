from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

login(os.environ["HUGGINGFACE_API_KEY"])

model_name = "meta-llama/Llama-3.3-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True, 
)