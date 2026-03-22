import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= PATH FIX =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

FAISS_PATH = os.path.join(BASE_DIR, "faiss_index", "index.faiss")
CLAIMS_PATH = os.path.join(BASE_DIR, "faiss_index", "claims.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "faiss_index", "labels.pkl")

MODEL_NAME = "intfloat/multilingual-e5-small"
print("FAISS PATH:", FAISS_PATH)
class SimilaritySearchEngine:
    def __init__(self):
        print("Loading FAISS index...")
        self.index = faiss.read_index(FAISS_PATH)
        print("FAISS loaded!")

        print("Loading claims and labels...")
        with open(CLAIMS_PATH, "rb") as f:
            self.claims = pickle.load(f)

        with open(LABELS_PATH, "rb") as f:
            self.labels = pickle.load(f)

        print("Data loaded!")

        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        print("Model loaded!")

    def search_evidence(self, query, top_k=5):
        query_prefixed = "query: " + query

        query_embedding = self.model.encode(
            [query_prefixed],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "claim": self.claims[idx],
                "label": self.labels[idx],
                "score": float(score)
            })

        return results