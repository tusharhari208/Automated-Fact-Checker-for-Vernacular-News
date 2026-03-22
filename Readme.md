**AUTOMATED FACT DETECTOR FOR VERNACULAR NEWS**

*Intel Unnati Industrial Training Project*

<p align="center">
  <img src="https://img.shields.io/badge/Intel-Unnati-0071C5?style=for-the-badge&logo=intel&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/RAG-Pipeline-00C853?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Live%20Demo-🟢-success?style=for-the-badge"/>
</p>

***

## Overview

AI-powered fact-checking system verifying news claims using a hybrid pipeline:

- **Classical ML** (99.94% accuracy classifiers)
- **FAISS semantic search** (145K FEVER embeddings)
- **Zero-shot NLI** (BERT-large-mnli)
- **Wikipedia fallback** + **Gemini LLM explanations**

**Outputs:**  REAL |  FAKE |  UNVERIFIED + human-readable reasoning 



## 🏗️ Architecture Pipeline

```
User Claim (Hindi/English) 
    ↓ [langdetect + deep-translator]
Translated Claim 
    ↓ [ScaleDown compression]
FAISS Top-5 Evidence (multilingual-e5-small)
    ↓ [BERT-large-mnli NLI]
Primary Verdict
    ↓ [Low confidence? → Wikipedia API]
Wikipedia Validation
    ↓ [Gemini-1.5-flash]
Final Verdict + Explanation
```

***

## ✨ Key Techniques

| Layer | Tech | Purpose |
|-------|------|---------|
| **Embeddings** | multilingual-e5-small | 384-dim vectors, 100+ languages  [ngrok](https://ngrok.com/docs/using-ngrok-with/fastAPI) |
| **Vector DB** | FAISS IndexFlatIP | 145K claims, dot-product similarity |
| **NLI** | BART-large-mnli | Entailment/Contradiction/Neutral |
| **ML Backup** | PassiveAggressiveClassifier | 99.94% accuracy on Kaggle dataset |
| **Fallbacks** | Wikipedia + Gemini | Handles novel claims |

***

##  Performance Metrics

| Metric | Value |
|--------|-------|
| **Dataset** | 145K FEVER + 44.9K Kaggle |
| **Latency** | 1-3s per claim |
| **Token Reduction** | 40-60% (ScaleDown API) |
| **Multilingual** | Hindi + 100 languages |
| **Throughput** | 20+ posts/minute |

***

##  Live Demo

**Public URL** (via pyngrok + uvicorn):  
`https://thelma-casketlike-cully.ngrok-free.dev/docs

**Test it now:**
```
POST /factcheck
{
  "claim": "Earth is flat"
}
```

**Response:**
```json
{
  "verdict": "FAKE",
  "confidence": 0.98,
  "explanation": "Scientific evidence confirms Earth is spherical...",
  "evidence": ["NASA images...", "Wikipedia: Spherical Earth"]
}
```

*URL active only when `uvicorn app:app --reload` + pyngrok running* [gist.github](https://gist.github.com/alexdlaird/1f0e45d943120d9e9cea8485773a0e4f)


fake_news_detection/
│
├── pipeline/
│     └── main_pipeline.py          # Main fact-checking pipeline
│     |---app.py
|      |--benchmark.py
├── src/
│   └── verification/
│       └── similarity_search.py  # FAISS semantic search engine
│
├── training/
│   └── train_classical.py        # ML model training script
│
├── prepare_data.py               # Builds FAISS index       
│
├── faiss_index/
│   ├── index.faiss               # Generated FAISS index     
│   ├── claims.pkl                # Generated claims          
│   ├── labels.pkl                # Generated labels          
│   ├── evidence.pkl              # Generated evidence       
│   └── embeddings.npy            # Generated embeddings      
│
├── saved_models/
│   ├── tfidf.pkl                 # Trained TF-IDF         
│   ├── passive_aggressive.pkl    # Trained PA model          
│   └── logistic_regression.pkl   # Trained LR model          
│
├── data/
│   └── raw/
│       ├── fake.csv              # Kaggle dataset            
│       └── true.csv              # Kaggle dataset            
│
├── requirements.txt              # All dependencies         
├── .env                          # Your API key              
├── .env.example                  # API key format           
├── .gitignore                    # Files to ignore           
└── README.md                     # Project documentation     

##  Quick Start

```bash
# Clone & setup
git clone <your-repo>
cd Fake_News_Detector
pip install -r requirements.txt

# Add API keys
echo "GEMINI_API_KEY=your_key" > .env
echo "SCALEDOWN_API_KEY=your_key" >> .env

# 1. Train models (one-time)
python train.py

# 2. Run API + Public URL
uvicorn app:app --reload --port 8000

# In another terminal: Start ngrok
python -c "
from pyngrok import ngrok
ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
"
```

**Postman test:** Send claims to your ngrok URL! [stackoverflow](https://stackoverflow.com/questions/63833593/how-to-run-fastapi-uvicorn-in-google-colab)

***

##  Tech Stack

```
API: FastAPI + uvicorn + pyngrok
AI: BART-mnli + multilingual-e5-small
ML: scikit-learn (99.94% acc) + FAISS
Data: FEVER (145K) + Kaggle (44.9K)
Utils: ScaleDown + Wikipedia + Gemini
```

***

##  Deployment Notes

- **Local:** `uvicorn app:app --reload` + pyngrok
- **Production:** Render/Railway (free tier) – auto-deploys from GitHub


**Keep alive:** URL expires when `app.py` stops [ngrok](https://ngrok.com/docs/using-ngrok-with/fastAPI)

***

##  Limitations & Future Work

- ✅ **Done:** Multilingual, explainable, production API
- 🔄 **Next:** Streamlit UI, Docker, persistent deployment



