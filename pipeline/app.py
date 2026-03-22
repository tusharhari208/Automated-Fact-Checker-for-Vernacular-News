from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import concurrent.futures
import time
import sys
import os
from dotenv import load_dotenv
from pyngrok import ngrok

# ================= LOAD ENV =================
load_dotenv()

NGROK_TOKEN = os.getenv("NGROK_TOKEN")

# 🔐 ONLY set if available
if NGROK_TOKEN:
    ngrok.set_auth_token(NGROK_TOKEN)

# ================= IMPORT PIPELINE =================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main_pipeline import FactChecker

# ================= APP SETUP =================
app = FastAPI(
    title="Automated Fact Checker",
    description="Fact Checker for Vernacular News — Intel Unnati",
    version="1.0.0"
)

# ================= LOAD MODEL =================
print("Loading FactChecker...")
fc = FactChecker()
print("FactChecker ready!")

# ================= REQUEST MODELS =================
class ClaimRequest(BaseModel):
    claim: str
    true_label: Optional[str] = None

class BatchRequest(BaseModel):
    claims: List[str]

# ================= HEALTH =================
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "ML + NLI + Wikipedia + Gemini",
        "version": "1.0.0"
    }

# ================= SINGLE =================
@app.post("/fact-check")
def check_single(request: ClaimRequest):
    try:
        result = fc.check(request.claim, request.true_label)

        return {
            "status": "success",
            "claim": request.claim,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "source": result["source"],
            "evidence": result["evidence"],
            "explanation": result["explanation"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================= BATCH =================
@app.post("/fact-check/batch")
def check_batch(request: BatchRequest):

    if len(request.claims) > 100:
        raise HTTPException(
            status_code=400,
            detail="Max 100 claims per batch"
        )

    try:
        start = time.time()
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:

            futures = {
                executor.submit(fc.check, claim): claim
                for claim in request.claims
            }

            for future in concurrent.futures.as_completed(futures):
                claim = futures[future]

                try:
                    result = future.result()

                    results.append({
                        "claim": claim,
                        "verdict": result["verdict"],
                        "confidence": result["confidence"],
                        "source": result["source"],
                        "evidence": result["evidence"],
                        "explanation": result["explanation"]
                    })

                except Exception as e:
                    results.append({
                        "claim": claim,
                        "verdict": "ERROR",
                        "error": str(e)
                    })

        elapsed = time.time() - start

        return {
            "status": "success",
            "total": len(request.claims),
            "elapsed_seconds": round(elapsed, 2),
            "throughput": round(len(request.claims) / elapsed * 60, 1),
            "unit": "posts/minute",
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================= EVALUATION =================
@app.get("/evaluate")
def evaluate():
    try:
        if len(fc.y_true) == 0:
            return {
                "status": "no data",
                "message": "Provide true_label in requests"
            }

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        return {
            "status": "success",
            "total": len(fc.y_true),
            "accuracy": round(accuracy_score(fc.y_true, fc.y_pred) * 100, 2),
            "precision": round(precision_score(fc.y_true, fc.y_pred, average='macro', zero_division=0) * 100, 2),
            "recall": round(recall_score(fc.y_true, fc.y_pred, average='macro', zero_division=0) * 100, 2),
            "f1_score": round(f1_score(fc.y_true, fc.y_pred, average='macro', zero_division=0) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================= RUN WITH NGROK =================
if __name__ == "__main__":

    load_dotenv()

    # Step 1 — set token
    ngrok.set_auth_token(os.getenv("NGROK_TOKEN"))

    # Step 2 — create tunnel BEFORE uvicorn
    public_url = ngrok.connect(8000)

    # Step 3 — print URLs
    print("\n" + "="*60)
    print("  FACT CHECKER API RUNNING!")
    print("="*60)
    print(f"  LOCAL URL  : http://localhost:8000")
    print(f"  PUBLIC URL : {public_url}")
    print("="*60)
    print("  Share PUBLIC URL with judges!")
    print("  They can test from anywhere!")
    print("="*60)
    print("  Endpoints:")
    print("  GET  /health")
    print("  POST /fact-check")
    print("  POST /fact-check/batch")
    print("  GET  /evaluate")
    print("="*60)

    # Step 4 — start server AFTER ngrok
    uvicorn.run(
        "app:app",
        host  = "0.0.0.0",
        port  = 8000,
        reload= False      # must be False with ngrok
    )


if __name__ == "__main__":

    print("\n" + "="*60)
    print("🚀 FACT CHECKER API STARTING")
    print("="*60)

    public_url = None

    # 🌍 Only create ngrok if token exists
    if NGROK_TOKEN:
        public_url = ngrok.connect(8000)
        print(f"🌍 PUBLIC URL : {public_url}")
    else:
        print("⚠️ NGROK NOT ENABLED (no token found)")

    print(f"💻 LOCAL URL  : http://127.0.0.1:8000")
    print(f"📄 DOCS       : http://127.0.0.1:8000/docs")

    print("="*60)

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )