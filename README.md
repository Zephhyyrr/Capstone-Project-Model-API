# API Speech-to-Text dan Koreksi Grammar Menggunakan Whisper AI dan LLAMA Secara Local

Proyek ini adalah API berbasis Flask yang menyediakan dua fungsi utama:

1. **Speech-to-Text**: Mengubah suara (file audio) menjadi teks menggunakan model Whisper.
2. **Koreksi Tata Bahasa**: Menganalisis dan mengoreksi tata bahasa Inggris dalam teks menggunakan model LLaMA.

API ini menerima file audio dan input teks, memprosesnya, lalu mengembalikan output dengan koreksi yang rinci.

## Fitur

- **Speech-to-Text**: Unggah file audio dan akan ditranskripsi menjadi teks menggunakan model Whisper.
- **Koreksi Tata Bahasa**: Kirimkan teks, dan sistem akan menganalisis serta mengoreksi kesalahan tata bahasa. Penjelasan perubahan juga disediakan.

## Requirement

Sebelum menjalankan aplikasi, pastikan paket-paket Python berikut telah diinstal:

- flask
- torch
- whisper
- nltk
- transformers
- werkzeug

Untuk menginstal paket yang dibutuhkan:

```bash
pip install requirement.tzt
```

## Preperation:

### Download Model ke Local

- **Whisper**: Digunakan untuk transkripsi suara ke teks.
- **LLaMA**: Digunakan untuk koreksi tata bahasa. Pastikan file model sudah diunduh dan ditempatkan pada direktori yang sesuai.

### Set up environment:

Konfigurasikan direktori cache Hugging Face dan folder untuk menyimpan audio yang diunggah dengan mengubah nilai berikut di dalam skrip:

- `HF_HOME`: Jalur penyimpanan model dari Hugging Face.
- `UPLOAD_FOLDER`: Jalur penyimpanan file audio yang diunggah.

### Buat Direktori yang Diperlukan

Pastikan direktori `uploads` tersedia untuk menyimpan file audio yang diunggah:

```bash
mkdir uploads
```

## Endpoint API

### 1. Speech-to-Text
Mengubah file audio menjadi teks.

- **URL**: `/speech-to-text`
- **Metode**: `POST`
- **Form Data**:
  - `audio`: File audio yang ingin ditranskripsi.

**Respon**:
```json
{
  "text": "Ini adalah teks hasil transkripsi dari audio."
}
```

**Contoh Penggunaan**:
```bash
curl -X POST -F "audio=@path_ke_file_audio.mp3" http://localhost:5051/speech-to-text
```

### 2. Koreksi Tata Bahasa
Menganalisis dan mengoreksi teks berbahasa Inggris.

- **URL**: `/analyze`
- **Metode**: `POST`
- **Form Data**:
  - `text`: Teks yang ingin dianalisis dan dikoreksi.

**Respon**:
```json
{
  "corrected_paragraph": "This is an example of an incorrect sentence.",
  "grammar_analysis": [
    {
      "original": "This are an example of incorrect sentence.",
      "corrected": "This is an example of an incorrect sentence.",
      "reason": "Corrected 'are' → 'is' for subject-verb agreement."
    }
  ]
}
```

**Contoh Penggunaan**:
```bash
curl -X POST -F "text=This are an example of incorrect sentence." http://localhost:5051/analyze
```

## Cara Kerja

- **Speech-to-Text**: API akan memverifikasi format file audio yang diunggah (seperti mp3, wav, dll). Lalu akan ditranskripsi menggunakan model Whisper dan mengembalikan hasil dalam bentuk teks.
- **Koreksi Tata Bahasa**: Teks dipecah menjadi kalimat menggunakan `nltk`, lalu diproses menggunakan model LLaMA. Koreksi tata bahasa dilakukan berdasarkan aturan seperti tenses, articles, preposisi, dll. Perbedaan antara kalimat asli dan yang telah dikoreksi akan disertai penjelasan singkat.

## Menjalankan Aplikasi

Untuk menjalankan Flask app secara lokal:

```bash
python app.py
```

Secara default, aplikasi berjalan di `http://localhost:5051`.
