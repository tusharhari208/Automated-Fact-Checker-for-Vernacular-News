import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = {
    "fake_path"  : os.path.join(BASE_DIR, "data", "raw", "fake.csv"),
    "true_path"  : os.path.join(BASE_DIR, "data", "raw", "true.csv"),
    "model_save" : os.path.join(BASE_DIR, "saved_models"),
    "test_size"  : 0.2,
}

# CREATE FOLDERS
os.makedirs(CONFIG["model_save"], exist_ok=True)

# STEP 1: LOAD DATASETS
print("Loading datasets...")

fake_df = pd.read_csv(CONFIG["fake_path"])
true_df = pd.read_csv(CONFIG["true_path"])

fake_df["label"] = 0  # 0 = FAKE
true_df["label"] = 1  # 1 = REAL

print("Fake news: " + str(len(fake_df)))
print("Real news: " + str(len(true_df)))

# STEP 2: COMBINE AND CLEAN
print("Combining datasets...")

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

# Combine title and text
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

print("Total articles: " + str(len(df)))

# STEP 3: SPLIT DATA
print("Splitting data...")

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = CONFIG["test_size"],
    random_state = 42
)

print("Training set: " + str(len(X_train)))
print("Testing set : " + str(len(X_test)))

# STEP 4: VECTORIZE (TF-IDF)
print("Vectorizing text...")

tfidf = TfidfVectorizer(
    max_features = 50000,
    ngram_range  = (1, 2),
    stop_words   = "english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print("Vectorization done!")

# STEP 5: TRAIN PASSIVE AGGRESSIVE
print("Training Passive Aggressive...")

pa_model = PassiveAggressiveClassifier(
    max_iter   = 1000,
    random_state = 42
)
pa_model.fit(X_train_tfidf, y_train)

pa_pred     = pa_model.predict(X_test_tfidf)
pa_accuracy = accuracy_score(y_test, pa_pred)

print("PA Accuracy: " + str(round(pa_accuracy * 100, 2)) + "%")
print(classification_report(y_test, pa_pred))

# STEP 6: TRAIN LOGISTIC REGRESSION
print("Training Logistic Regression...")

lr_model = LogisticRegression(
    max_iter     = 1000,
    random_state = 42
)
lr_model.fit(X_train_tfidf, y_train)

lr_pred     = lr_model.predict(X_test_tfidf)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("LR Accuracy: " + str(round(lr_accuracy * 100, 2)) + "%")
print(classification_report(y_test, lr_pred))

# STEP 7: SAVE ALL MODELS
print("Saving models...")

# Save TF-IDF
tfidf_path = os.path.join(CONFIG["model_save"], "tfidf.pkl")
with open(tfidf_path, "wb") as f:
    pickle.dump(tfidf, f)
print("tfidf.pkl saved!")

# Save PA model
pa_path = os.path.join(CONFIG["model_save"], "passive_aggressive.pkl")
with open(pa_path, "wb") as f:
    pickle.dump(pa_model, f)
print("passive_aggressive.pkl saved!")

# Save LR model
lr_path = os.path.join(CONFIG["model_save"], "logistic_regression.pkl")
with open(lr_path, "wb") as f:
    pickle.dump(lr_model, f)
print("logistic_regression.pkl saved!")

print("All models saved in saved_models/!")
print("PA Accuracy: " + str(round(pa_accuracy * 100, 2)) + "%")
print("LR Accuracy: " + str(round(lr_accuracy * 100, 2)) + "%")