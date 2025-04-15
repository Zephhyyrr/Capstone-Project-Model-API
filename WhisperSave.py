import whisper

# Path untuk menyimpan model Whisper
model_path = model_path = r"D:\Laporan Kuliah Semester 6\Capstone Project\models"

# Load model Whisper (bisa ganti dengan "small", "medium", atau "large")
whisper_model = whisper.load_model("turbo", download_root=model_path)

print("✅ Model Whisper berhasil diunduh dan dimuat!")
