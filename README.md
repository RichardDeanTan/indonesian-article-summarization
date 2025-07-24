# 📰 Indonesian Article Summarizer - Fine-tuned IndoBART Model

Proyek ini menggunakan fine-tuned IndoBART model untuk menghasilkan ringkasan otomatis dari artikel bahasa Indonesia. Model ini telah di-train khusus untuk memahami struktur dan konteks bahasa Indonesia, menghasilkan ringkasan yang akurat dan relevan.

## 📂 Project Structure

- `IndoArticle (Baseline).ipynb` — Jupyter Notebook untuk baseline model dan initial experiments.
- `IndoArticle (Save Model).ipynb` — Jupyter Notebook untuk menyimpan model yang sudah di-train.
- `IndoArticle (Tuning).ipynb` — Jupyter Notebook untuk fine-tuning dan optimasi model.
- `IndoData200.csv` — Dataset artikel bahasa Indonesia untuk training dan evaluasi.
- `app.py` — Aplikasi Streamlit untuk interface web prediksi.
- `requirements.txt` — Daftar dependensi Python yang diperlukan untuk menjalankan project.

## 🚀 Cara Run Aplikasi

### 🔹 1. Jalankan Secara Lokal
### Clone Repository
```bash
git clone https://github.com/RichardDeanTan/Indonesian-Article-Summarization.git
cd Indonesian-Article-Summarization
```
### Install Dependensi
```bash
pip install -r requirements.txt
```
### Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

### 🔹 2. Jalankan Secara Online (Tidak Perlu Install)
Klik link berikut untuk langsung membuka aplikasi web:
#### 👉 [Streamlit - Indonesian Article Summarizer](https://indonesian-article-summarization-richardtanjaya.streamlit.app/)

Klik link berikut untuk Model fine-tuned tersedia di Hugging Face Hub:
#### 👉 [huggingface - RichTan/indobart-custom](https://huggingface.co/RichTan/indobart-custom)

## 💡 Fitur
- ✅ **Automatic Summarization:** Menghasilkan ringkasan otomatis dari artikel bahasa Indonesia
- ✅ **Interactive Web Interface:** Interface yang user-friendly menggunakan Streamlit
- ✅ **Real-time Processing:** Prediksi ringkasan secara real-time
- ✅ **Customizable Parameters:** Mengatur panjang ringkasan dan parameter beam search
- ✅ **Model Metrics:** Menampilkan ROUGE scores dan informasi model
- ✅ **Download Feature:** Download hasil ringkasan dalam format .txt
- ✅ **Example Articles:** Contoh artikel untuk testing aplikasi

## ⚙️ Tech Stack
- **Deep Learning:** PyTorch, Transformers (Hugging Face)
- **Model:** Fine-tuned IndoBART
- **Data Processing:** Pandas, NumPy
- **Model Hub:** Hugging Face Hub
- **Deployment:** Streamlit Cloud

## 🧠 Model Details
- **Base Model:** IndoBART (Indonesian BART)
- **Task:** Text Summarization
- **Language:** Indonesian (Bahasa Indonesia)
- **Fine-tuning:** Custom dataset dengan 200 artikel Indonesia
- **Evaluation Metrics:** ROUGE-1, ROUGE-2, ROUGE-L scores

## ⭐ Deployment
Aplikasi ini di-deploy menggunakan:
- Streamlit Cloud
- Hugging Face Hub
- GitHub

## 👨‍💻 Pembuat
Richard Dean Tanjaya

## 📝 License
Proyek ini bersifat open-source dan bebas digunakan untuk keperluan edukasi dan penelitian.