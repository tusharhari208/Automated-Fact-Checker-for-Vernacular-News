from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from main_pipeline import fact_check

app = FastAPI(
    title      = "Fact Checker API",
    description= "Automated Fact-Checker for Vernacular News",
    version    = "1.0.0"
)

# ================= REQUEST MODELS =================
class ClaimRequest(BaseModel):
    claim: str

class BatchRequest(BaseModel):
    claims: List[str]

# ================= SINGLE CHECK =================
@app.post("/fact-check")
def check_single(request: ClaimRequest):
    try:
        result = fact_check(request.claim)
        return {
            "status" : "success",
            "claim"  : request.claim,
            "result" : result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================= BATCH CHECK =================
@app.post("/fact-check/batch")
def check_batch(request: BatchRequest):
    if len(request.claims) > 100:
        raise HTTPException(
            status_code = 400,
            detail      = "Max 100 claims per batch"
        )
    try:
        import concurrent.futures
        import time

        start   = time.time()
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(fact_check, claim): claim
                for claim in request.claims
            }
            for future in concurrent.futures.as_completed(futures):
                claim  = futures[future]
                result = future.result()
                results.append({
                    "claim" : claim,
                    "result": result
                })

        elapsed    = time.time() - start
        throughput = len(request.claims) / elapsed * 60

        return {
            "status"    : "success",
            "total"     : len(request.claims),
            "elapsed_s" : round(elapsed, 2),
            "throughput": round(throughput, 1),
            "unit"      : "posts/minute",
            "results"   : results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================= HEALTH CHECK =================
@app.get("/health")
def health():
    return {"status": "running"}

# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)