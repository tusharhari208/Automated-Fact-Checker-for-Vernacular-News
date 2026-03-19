import numpy as np
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

class SimilaritySearchEngine:

    def __init__(self):
        print("Loading similarity search engine...")

        # Fix paths
        BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        index_path    = os.path.join(BASE_DIR, "faiss_index", "index.faiss")
        claims_path   = os.path.join(BASE_DIR, "faiss_index", "claims.pkl")
        labels_path   = os.path.join(BASE_DIR, "faiss_index", "labels.pkl")
        evidence_path = os.path.join(BASE_DIR, "faiss_index", "evidence.pkl")

        # Load FAISS index
        try:
            self.index = faiss.read_index(index_path)
        except Exception as e:
            raise FileNotFoundError("FAISS index not found! Run build_index.py first.\n" + str(e))
        print("FAISS index loaded!")
        print("Total claims indexed: " + str(self.index.ntotal))

        # Load claims
        with open(claims_path, "rb") as f:
            self.claims = pickle.load(f)
        print("Claims loaded: " + str(len(self.claims)))

        # Load labels
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)
        print("Labels loaded!")

        # Load evidence
        with open(evidence_path, "rb") as f:
            self.evidence = pickle.load(f)
        print("Evidence loaded!")

        # Load embedding model
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")
        print("Embedding model loaded!")
        print("Search engine ready!")

    def search(self, query, top_k=5):
        query_prefixed  = "query: " + query
        query_embedding = self.model.encode(
            [query_prefixed],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "claim"    : self.claims[idx],
                "label"    : self.labels[idx],
                "score"    : float(score),
                "evidence" : self.evidence[idx]
            })
        return results

    def get_verdict(self, query):
        results = self.search(query, top_k=5)

        print("Top 5 Similar Claims Found:")
        print("=" * 50)

        supports_count = 0
        refutes_count  = 0
        nei_count      = 0

        for i, result in enumerate(results, 1):
            print("Rank " + str(i) + ":")
            print("  Score : " + str(round(result["score"], 4)))
            print("  Label : " + result["label"])
            print("  Claim : " + result["claim"][:60] + "...")
            print()

            if result["label"] == "SUPPORTS":
                supports_count += 1
            elif result["label"] == "REFUTES":
                refutes_count += 1
            else:
                nei_count += 1

        print("Label Counts:")
        print("  SUPPORTS       : " + str(supports_count))
        print("  REFUTES        : " + str(refutes_count))
        print("  NOT ENOUGH INFO: " + str(nei_count))

        if supports_count > refutes_count:
            verdict    = "REAL"
            confidence = (supports_count / 5) * 100
            reason     = str(supports_count) + "/5 claims SUPPORT this"

        elif refutes_count > supports_count:
            verdict    = "FAKE"
            confidence = (refutes_count / 5) * 100
            reason     = str(refutes_count) + "/5 claims REFUTE this"

        else:
            verdict    = "UNVERIFIED"
            # Fix: 0% confidence if ALL are NEI, else 50%
            confidence = 0 if (supports_count == 0 and refutes_count == 0) else 50
            reason     = "Not enough evidence found"

        return {
            "verdict"    : verdict,
            "confidence" : confidence,
            "reason"     : reason,
            "top_results": results
        }


# TEST THE ENGINE
if __name__ == "__main__":

    print("Testing Similarity Search Engine...")
    print("=" * 50)

    engine = SimilaritySearchEngine()

    test_claims = [
        "Vaccines cause serious side effects",
        "The earth is flat",
        "NASA successfully landed on moon",
    ]

    for claim in test_claims:
        print("\nCLAIM: " + claim)
        print("-" * 50)

        verdict = engine.get_verdict(claim)

        print("VERDICT    : " + verdict["verdict"])
        print("CONFIDENCE : " + str(verdict["confidence"]) + "%")
        print("REASON     : " + verdict["reason"])
        print("=" * 50)