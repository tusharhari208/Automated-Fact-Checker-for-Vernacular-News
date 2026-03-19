# 🛡️ Fake News Detector
### Intel Unnati Industrial Training Project

<p align="center">
  <img src="https://img.shields.io/badge/Intel-Unnati-0071C5?style=for-the-badge&logo=intel&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google-Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/RAG-Pipeline-00C853?style=for-the-badge"/>
</p>

---

## 📌 Overview

**Fake News Detector** is an AI-powered fact-checking system built as part of the **Intel Unnati Industrial Training Program**. It uses a hybrid approach combining **Classical Machine Learning**, **Semantic Search**, **Wikipedia Live Search**, and **Google Gemini LLM** to detect whether a given news article is **REAL**, **FAKE**, or **UNVERIFIED** — with a human-readable explanation.

> 💡 The system supports **multilingual news** using the `multilingual-e5-small` embedding model, making it capable of detecting fake news across multiple languages.

---

## 🎯 Problem Statement

With the rapid spread of misinformation on the internet, it has become increasingly difficult to distinguish real news from fake news. This project addresses that challenge by building an end-to-end automated fact-checking pipeline that:

- Detects fake news using **ML models trained on real-world datasets**
- Verifies claims against **100,000+ verified facts** from the FEVER dataset
- Cross-checks with **live Wikipedia** for out-of-distribution news
- Generates a **clear, human-readable explanation** using Google Gemini LLM

---

## 🏗️ System Architecture

```
                        USER INPUT (News Article)
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
  ┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │   JUDGE 1     │     │    JUDGE 2      │     │    JUDGE 3      │
  │   Passive     │     │   Logistic      │     │  FAISS Semantic │
  │  Aggressive   │     │  Regression     │     │     Search      │
  │   Model       │     │   Model         │     │  (FEVER Dataset)│
  └───────┬───────┘     └───────┬─────────┘     └───────┬─────────┘
          │                     │                        │
          └─────────────────────┼────────────────────────┘
                                │
                       ┌────────▼────────┐
                       │    JUDGE 4      │
                       │   Wikipedia     │
                       │  Live Search    │
                       └────────┬────────┘
                                │
                    ┌───────────▼───────────┐
                    │   SCORING SYSTEM      │
                    │  REAL    = +1 point   │
                    │  FAKE    = -1 point   │
                    │  UNKNOWN =  0 points  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   RAG + GEMINI LLM    │
                    │  Retrieval Augmented  │
                    │  Generation Pipeline  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │     FINAL VERDICT     │
                    │  ✅ REAL / ❌ FAKE    │
                    │  ⚠️ UNVERIFIED        │
                    │  + LLM Explanation    │
                    └───────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| 🤖 **4-Judge Voting System** | PA Model + LR Model + FAISS Search + Wikipedia |
| 🌍 **Multilingual Support** | Detects fake news in multiple languages |
| 🔍 **Semantic Search** | FAISS index with 100,000+ FEVER verified claims |
| 📡 **Live Wikipedia Check** | Handles out-of-distribution/new news |
| 🧠 **RAG Pipeline** | Retrieval Augmented Generation for explanations |
| 💬 **LLM Explanation** | Google Gemini generates human-readable verdict |
| 📰 **Batch Processing** | Check multiple news articles at once |
| 🔒 **Secure API Handling** | API keys stored safely using `.env` |

---

## 🧠 RAG Pipeline Explained

```
RETRIEVE  →  FAISS finds Top 5 similar verified claims
             Wikipedia finds related article summary

AUGMENT   →  All evidence is added as context to prompt

GENERATE  →  Google Gemini LLM reads all evidence and
             generates a clear, human-readable explanation
```

---

## 📂 Project Structure

```
fake_news_detection/
│
├── pipeline/
│   └── main_pipeline.py          # Main fact-checking pipeline
│
├── src/
│   └── verification/
│       └── similarity_search.py  # FAISS semantic search engine
│
├── training/
│   └── train_classical.py        # ML model training script
│
├── prepare_data.py               # Builds FAISS index        ✅ upload
│
├── faiss_index/
│   ├── index.faiss               # Generated FAISS index     ❌ don't upload
│   ├── claims.pkl                # Generated claims          ❌ don't upload
│   ├── labels.pkl                # Generated labels          ❌ don't upload
│   ├── evidence.pkl              # Generated evidence        ❌ don't upload
│   └── embeddings.npy            # Generated embeddings      ❌ don't upload
│
├── saved_models/
│   ├── tfidf.pkl                 # Trained TF-IDF            ❌ don't upload
│   ├── passive_aggressive.pkl    # Trained PA model          ❌ don't upload
│   └── logistic_regression.pkl   # Trained LR model          ❌ don't upload
│
├── data/
│   └── raw/
│       ├── fake.csv              # Kaggle dataset            ❌ don't upload
│       └── true.csv              # Kaggle dataset            ❌ don't upload
│
├── requirements.txt              # All dependencies          ✅ upload
├── .env                          # Your API key              ❌ don't upload
├── .env.example                  # API key format            ✅ upload
├── .gitignore                    # Files to ignore           ✅ upload
└── README.md                     # Project documentation     ✅ upload
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/tusharhari208/fake_news_detection.git
cd fake_news_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Datasets
- Download `fake.csv` and `true.csv` from [Kaggle - Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Place them in `data/raw/`

### 4. Set Up API Key
- Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com)
- Create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

### 5. Train Classical Models
```bash
# Trains PA + LR models and saves them in saved_models/
python training/train_classical.py
```

### 6. Build FAISS Index
```bash
# Loads FEVER dataset, converts claims to embeddings
# using multilingual-e5-small and saves FAISS index
```

### 7. Run the Pipeline
```bash
python pipeline/main_pipeline.py
```

> ⚠️ **Note:** Steps 5 and 6 are **one-time setup steps**. Once models and FAISS index are built, you only need to run Step 7 every time.

---

## 💻 How to Use

```
==============================
      FAKE NEWS DETECTOR
FORMAT:
  1. News article one here
  2. News article two here
Press ENTER TWICE to submit.
Type 'quit' to exit.
==============================

Paste your numbered news below:
1. Scientists say 5G towers cause coronavirus infections
2. NASA successfully landed astronauts on the moon in 1969

Found 2 articles to check!

>>> ARTICLE 1 of 2
Stage 1 - Classical Models:
  Passive Aggressive : FAKE
  Logistic Regression: FAKE
Stage 2 - Similarity Search...
  Evidence Verdict   : FAKE
Stage 3 - Wikipedia Check...
  Wikipedia Verdict  : NOT SUPPORTED
Stage 4 - Final Verdict: FAKE
Stage 5 - Generating LLM Explanation...

FINAL VERDICT   : ❌ FAKE
LLM EXPLANATION :
This news is fake. According to Wikipedia and multiple
verified claims from the FEVER dataset, there is no
scientific evidence linking 5G towers to coronavirus.
The claim has been widely debunked by health authorities
worldwide including WHO and CDC.
```

---

## 📊 Models Used

| Model | Purpose | Dataset |
|-------|---------|---------|
| `Passive Aggressive Classifier` | Fast fake news detection | Kaggle Fake/Real News |
| `Logistic Regression` | Fast fake news detection | Kaggle Fake/Real News |
| `multilingual-e5-small` | Text embeddings | HuggingFace |
| `FAISS Index` | Semantic similarity search | FEVER Dataset |
| `Google Gemini 1.5 Flash` | LLM explanation generation | Google AI |

---

## 📦 Requirements

```txt
numpy
pandas
scikit-learn
sentence-transformers
transformers
torch
faiss-cpu
datasets
wikipedia
requests
huggingface-hub
tqdm
google-generativeai
python-dotenv
```

---

## 🔒 Security

- API keys are stored in `.env` file which is **never uploaded to GitHub**
- `.env.example` is provided so others know the format
- All sensitive files are listed in `.gitignore`

---

## 👨‍💻 Built With

- [scikit-learn](https://scikit-learn.org/) — Classical ML Models
- [FAISS](https://github.com/facebookresearch/faiss) — Vector Similarity Search
- [Sentence Transformers](https://www.sbert.net/) — Multilingual Embeddings
- [FEVER Dataset](https://fever.ai/) — Verified Claims Database
- [Wikipedia API](https://pypi.org/project/wikipedia/) — Live Fact Checking
- [Google Gemini](https://aistudio.google.com/) — LLM Explanation

---

## 🏆 Intel Unnati Industrial Training

This project was developed as part of the **Intel Unnati Industrial Training Program** — a flagship initiative by Intel India to provide students with hands-on experience in cutting-edge technologies including AI, ML, and Deep Learning.

---

## ⚠️ Disclaimer

This tool is built for educational purposes as part of Intel Unnati training. While it uses multiple verification methods, no automated fact-checking system is 100% accurate. Always verify important news from trusted sources.

---


