# 🤖 ATS Resume Chatbot

> Match your resume to any job description, find the gaps, and get AI-powered suggestions — all locally, no API keys required.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![No API Key](https://img.shields.io/badge/API%20Key-Not%20Required-lightgrey?style=flat-square)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Match Scoring** | Cosine similarity using sentence embeddings |
| 🔍 **Keyword Extraction** | Identifies tech skills, phrases, and domain keywords |
| 🕳️ **Gap Analysis** | Shows exactly what you have vs. what's missing |
| 💡 **AI Suggestions** | Resume improvement suggestions via Llama |
| 💬 **Chat Interface** | Ask follow-up questions about your resume |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/mikayla-james/ATS_chatbot.git
cd ATS_chatbot

# 2. Create and activate a virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run ats_app.py
```

Opens at **http://localhost:8501**

> ⚠️ First launch downloads ~1.1 GB of models. Cached after that, so subsequent launches are fast.

---

## 💻 System Requirements

- Python 3.9+
- ~2 GB disk space
- 4+ GB RAM recommended
- CPU only — no GPU needed

---

## 🛠️ Customization Ideas

- 📋 **Google Sheets integration** via `gspread` to log applications
- 📄 **PDF upload** with `pdfplumber`
- 📦 **Batch analysis** against multiple job descriptions at once
- ☁️ **Free deployment** on Streamlit Cloud

---

## 🏗️ How It Works

```
Resume + Job Description
        ↓
Sentence Embeddings (all-MiniLM-L6-v2)
        ↓
Cosine Similarity Score + Gap Analysis
        ↓
Llama → Tailored Suggestions + Chat
```

---

## 📄 License

MIT — free to use, modify, and build on.
