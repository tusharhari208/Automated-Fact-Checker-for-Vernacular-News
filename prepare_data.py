import pandas as pd
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# CONFIGURATION
CONFIG = {
    "data_path"       : "data/raw/train.jsonl",
    "save_folder"     : "faiss_index",
    "processed_folder": "data/processed",
    "batch_size"      : 64,
    "model_name"      : "intfloat/multilingual-e5-small",
    "max_chars"       : 200,   # scaledown limit for FEVER claims
}

# ================= SCALE DOWN =================
def scale_down(text, max_chars=CONFIG["max_chars"]):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated   = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > 80:
        return truncated[:last_period + 1]
    return truncated

login(token="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")

# STEP 0: CREATE REQUIRED FOLDERS
print("Creating required folders...")
os.makedirs(CONFIG["save_folder"], exist_ok=True)
os.makedirs(CONFIG["processed_folder"], exist_ok=True)
print("Folders ready!")

# STEP 1: LOAD MODEL
print("Loading embedding model...")
model = SentenceTransformer(CONFIG["model_name"])
print("Model loaded!")

# STEP 2: LOAD DATASET
print("Loading FEVER dataset...")

if not os.path.exists(CONFIG["data_path"]):
    print("ERROR: Dataset not found!")
    print("Place train.jsonl inside data/raw/")
    exit()

data = pd.read_json(CONFIG["data_path"], lines=True)
print("Dataset loaded!")
print("Total rows: " + str(len(data)))
print("Columns   : " + str(list(data.columns)))

# STEP 3: EXTRACT + SCALE DOWN
print("Extracting and scaling down claims...")

raw_claims = data["claim"].tolist()
labels     = data["label"].tolist()

# apply scale down to claims
claims = [scale_down(c) for c in raw_claims]

print("Raw claims   : " + str(len(raw_claims)))
print("Scaled claims: " + str(len(claims)))
print("Labels found : " + str(set(labels)))
print("Sample raw   : " + raw_claims[0][:80])
print("Sample scaled: " + claims[0][:80])
print("Sample label : " + labels[0])

# check how much we reduced
avg_raw    = sum(len(c) for c in raw_claims) / len(raw_claims)
avg_scaled = sum(len(c) for c in claims)     / len(claims)
print("Avg chars before: " + str(round(avg_raw, 1)))
print("Avg chars after : " + str(round(avg_scaled, 1)))
print("Reduction       : " + str(round((1 - avg_scaled/avg_raw)*100, 1)) + "%")

if "evidence" in data.columns:
    evidence = data["evidence"].tolist()
    print("Evidence: " + str(len(evidence)))
else:
    evidence = [""] * len(claims)
    print("No evidence column, using empty")

# STEP 4: SAVE PROCESSED DATA
print("Saving processed data...")

processed_path = os.path.join(
    CONFIG["processed_folder"],
    "processed_claims.pkl"
)

processed_data = {
    "claims"  : claims,
    "labels"  : labels,
    "evidence": evidence
}

with open(processed_path, "wb") as f:
    pickle.dump(processed_data, f)

print("Saved to: " + processed_path)

# STEP 5: GENERATE EMBEDDINGS
print("Generating embeddings...")
print("Total claims: " + str(len(claims)))
print("Batch size  : " + str(CONFIG["batch_size"]))

checkpoint_path = os.path.join(
    CONFIG["save_folder"],
    "checkpoint.npy"
)

all_embeddings = []
start_idx      = 0

if os.path.exists(checkpoint_path):
    print("Checkpoint found! Resuming...")
    saved_checkpoint = np.load(checkpoint_path)
    all_embeddings   = [saved_checkpoint]
    start_idx        = len(saved_checkpoint)
    print("Resuming from claim: " + str(start_idx))
else:
    print("No checkpoint, starting fresh...")

batch_size = CONFIG["batch_size"]

for i in range(start_idx, len(claims), batch_size):

    batch = claims[i : i + batch_size]

    batch_prefixed = [
        "passage: " + claim
        for claim in batch
    ]

    embeddings_batch = model.encode(
        batch_prefixed,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    all_embeddings.append(embeddings_batch)

    processed  = min(i + batch_size, len(claims))
    percentage = (processed / len(claims)) * 100

    print(
        str(processed) + "/" +
        str(len(claims)) +
        " (" + str(round(percentage, 1)) + "%) done"
    )

    batch_number = i // batch_size
    if batch_number % 10 == 0 and i > 0:
        temp = np.vstack(all_embeddings)
        np.save(checkpoint_path, temp)
        print("Checkpoint saved at " + str(processed))

print("Combining all embeddings...")
embeddings = np.vstack(all_embeddings).astype("float32")
print("Shape: " + str(embeddings.shape))

# STEP 6: BUILD FAISS INDEX
print("Building FAISS index...")

dimension = embeddings.shape[1]
index     = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("FAISS index built!")
print("Vectors indexed: " + str(index.ntotal))

# STEP 7: SAVE ALL FILES
print("Saving all files...")

save = CONFIG["save_folder"]

faiss_path = os.path.join(save, "index.faiss")
faiss.write_index(index, faiss_path)
size = os.path.getsize(faiss_path) / (1024*1024)
print("index.faiss saved! " + str(round(size,1)) + " MB")

emb_path = os.path.join(save, "embeddings.npy")
np.save(emb_path, embeddings)
size = os.path.getsize(emb_path) / (1024*1024)
print("embeddings.npy saved! " + str(round(size,1)) + " MB")

claims_path = os.path.join(save, "claims.pkl")
with open(claims_path, "wb") as f:
    pickle.dump(claims, f)
size = os.path.getsize(claims_path) / (1024*1024)
print("claims.pkl saved! " + str(round(size,1)) + " MB")

labels_path = os.path.join(save, "labels.pkl")
with open(labels_path, "wb") as f:
    pickle.dump(labels, f)
size = os.path.getsize(labels_path) / (1024*1024)
print("labels.pkl saved! " + str(round(size,1)) + " MB")

evidence_path = os.path.join(save, "evidence.pkl")
with open(evidence_path, "wb") as f:
    pickle.dump(evidence, f)
size = os.path.getsize(evidence_path) / (1024*1024)
print("evidence.pkl saved! " + str(round(size,1)) + " MB")

if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("Checkpoint deleted!")

# STEP 8: VERIFY FILES
print("Verifying saved files...")

files_to_verify = [
    "index.faiss",
    "embeddings.npy",
    "claims.pkl",
    "labels.pkl",
    "evidence.pkl"
]

all_good   = True
total_size = 0

for filename in files_to_verify:
    full_path = os.path.join(CONFIG["save_folder"], filename)
    if os.path.exists(full_path):
        size       = os.path.getsize(full_path)
        size_mb    = size / (1024 * 1024)
        total_size += size_mb
        print("OK: " + filename + " - " + str(round(size_mb,1)) + " MB")
    else:
        print("MISSING: " + filename)
        all_good = False

print("Total size: " + str(round(total_size,1)) + " MB")

if all_good:
    print("All files verified!")
else:
    print("Some files missing!")

# STEP 9: TEST SEARCH
print("Testing search function...")

def search_evidence(query, top_k=5):
    query_prefixed  = "query: " + scale_down(query)
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

test_query = "Vaccines are safe and effective"
print("Test query: " + test_query)

results = search_evidence(test_query, top_k=3)

for i, result in enumerate(results, 1):
    print("Result " + str(i) + ":")
    print("  Score: " + str(round(result["score"], 4)))
    print("  Label: " + result["label"])
    print("  Claim: " + result["claim"][:60] + "...")

print("\nprepare_data.py COMPLETE!")
print("Scale-down applied — faster embeddings!")
print("Files saved in faiss_index/ folder")
print("Next step: run main_pipeline.py")



