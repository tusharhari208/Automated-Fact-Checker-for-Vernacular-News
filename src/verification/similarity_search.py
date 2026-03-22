import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# CONFIG
FAISS_PATH = "faiss_index/index.faiss"
CLAIMS_PATH = "faiss_index/claims.pkl"
LABELS_PATH = "faiss_index/labels.pkl"
MODEL_NAME = "intfloat/multilingual-e5-small"

# ================= LOAD PREBUILT DATA =================
print("Loading FAISS index...")
index = faiss.read_index(FAISS_PATH)
print("FAISS loaded!")

print("Loading claims and labels...")
with open(CLAIMS_PATH, "rb") as f:
    claims = pickle.load(f)

with open(LABELS_PATH, "rb") as f:
    labels = pickle.load(f)

print("Data loaded!")

# ================= LOAD MODEL =================
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded!")

# ================= SEARCH FUNCTION =================
def search_evidence(query, top_k=5):
    query_prefixed = "query: " + query

    query_embedding = model.encode(
        [query_prefixed],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "claim": claims[idx],
            "label": labels[idx],
            "score": float(score)
        })

    return results

# ================= TEST =================
if __name__ == "__main__":
    test_query = "Vaccines are safe and effective"
    print("Test query:", test_query)

    results = search_evidence(test_query, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print("  Score:", round(result["score"], 4))
        print("  Label:", result["label"])
        print("  Claim:", result["claim"][:60] + "...")
