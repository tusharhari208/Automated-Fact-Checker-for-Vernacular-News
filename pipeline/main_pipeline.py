import os
import sys
import time
import pickle
import wikipedia
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import scaledown as sd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.verification.similarity_search import SimilaritySearchEngine

# Load env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
sd.set_api_key(os.getenv("SCALEDOWN_API_KEY"))

# ============================================
# FACT CHECKER - WIKIPEDIA HIGHEST PRIORITY
# ============================================
class FactChecker:
    def __init__(self):
        print("\n" + "="*70)
        print("  🚀 LOADING ADVANCED FACT CHECKER SYSTEM...")
        print("="*70)

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ML Models (Optional)
        try:
            with open(os.path.join(BASE_DIR, "saved_models", "tfidf.pkl"), "rb") as f:
                self.tfidf = pickle.load(f)
            with open(os.path.join(BASE_DIR, "saved_models", "logistic_regression.pkl"), "rb") as f:
                self.lr_model = pickle.load(f)
            self.use_ml = True
            print("  ✅ ML models loaded!")
        except:
            self.use_ml = False
            print("  ⚠️  ML models not found")

        # FAISS (Optional)
        self.search_engine = SimilaritySearchEngine()
        print("  ✅ FAISS loaded!")

        # NLI model
        print("  🔄 Loading NLI model...")
        self.nli_model = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
        print("  ✅ NLI loaded!")

        # Gemini LLM (Fallback)
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.llm = genai.GenerativeModel("gemini-1.5-flash")
            print("  ✅ Gemini ready!")
        else:
            self.llm = None
            print("  ⚠️  Gemini API key missing")

        # ScaleDown Compressor
        try:
            self.compressor = ScaleDownCompressor(target_model="gpt-4o", rate="auto")
            print("  ✅ ScaleDown ready!")
        except:
            self.compressor = None
            print("  ⚠️  ScaleDown unavailable")

        # Token counter
        self.token_count = lambda text: len(text.split())
        self.y_true = []
        self.y_pred = []

        print("="*70)
        print("  🎯 SYSTEM READY!")
        print("="*70)

    def translate(self, text):
        try:
            if detect(text) != "en":
                text = GoogleTranslator(source='auto', target='en').translate(text)
            return text.lower().strip()
        except:
            return text.lower().strip()

    def compress_claim(self, text, max_len=150):
        if not self.compressor:
            return text[:max_len]
        try:
            result = self.compressor.compress(context="", prompt="Extract core factual claim briefly: " + text)
            return result.prompt[:max_len]
        except:
            return text[:max_len]

    def count_tokens(self, text):
        return self.token_count(text)

    def check_wikipedia_priority(self, text):
        """WIKIPEDIA = HIGHEST PRIORITY CHECK"""
        print("📚 1️⃣ WIKIPEDIA VERIFICATION (Priority)")
        
        try:
            # Search multiple pages for best match
            res = wikipedia.search(text, results=5)
            if not res:
                print("   ❌ No Wikipedia pages found")
                return None, 0.0, ""
            
            print(f"   🔍 Found {len(res)} pages: {res[0]}")
            
            # Try first 3 pages deeply
            for i, page in enumerate(res[:3]):
                try:
                    summary = wikipedia.summary(page, sentences=4)
                    compressed_summary = self.compress_claim(summary)
                    
                    # NLI verification on Wikipedia content
                    nli_result = self.nli_model(
                        sequences=text,
                        candidate_labels=["entailment", "contradiction", "neutral"],
                        hypothesis_template="This claim {} the following evidence: {}.".format("", compressed_summary)
                    )
                    
                    top_label = nli_result["labels"][0]
                    top_score = nli_result["scores"][0] * 100
                    
                    print(f"   📖 Page {i+1}: {page[:40]}... | NLI: {top_label} ({top_score:.1f}%)")
                    
                    # Wikipedia success criteria
                    if top_label == "entailment" and top_score > 75:
                        return "TRUE", top_score, compressed_summary, page
                    elif top_label == "contradiction" and top_score > 75:
                        return "FALSE", top_score, compressed_summary, page
                        
                except Exception as e:
                    print(f"   ⚠️  Page {i+1} failed: {e}")
                    continue
            
            print("   ⚠️  Wikipedia inconclusive")
            return None, 0.0, ""
            
        except Exception as e:
            print(f"   ❌ Wikipedia error: {e}")
            return None, 0.0, ""

    def ml_check(self, text):
        if not self.use_ml:
            return "UNKNOWN", 0.0
        try:
            ml_features = self.tfidf.transform([text])
            ml_pred = self.lr_model.predict(ml_features)[0]
            ml_prob = self.lr_model.predict_proba(ml_features)[0]
            return ("TRUE" if ml_pred == 1 else "FALSE"), max(ml_prob) * 100
        except:
            return "UNKNOWN", 0.0

    def faiss_search(self, text):
        try:
            results = self.search_engine.search_evidence(text, top_k=3)
            strong = [r for r in results if r["score"] > 0.75]
            return strong[0]["claim"] if strong else None, strong[0]["score"] * 100 if strong else 0.0
        except:
            return None, 0.0

    def llm_fallback(self, text, wiki_evidence=""):
        if not self.llm:
            return "UNVERIFIABLE", 50.0, "LLM unavailable"
        
        prompt = f"""Verify this claim using Wikipedia evidence:

Claim: {text}
Wikipedia: {wiki_evidence}

VERDICT: TRUE/FALSE/UNVERIFIABLE
CONFIDENCE: [0-100]%
REASON: [one sentence]"""
        
        try:
            response = self.llm.generate_content(prompt)
            result = response.text
            
            if "TRUE" in result.upper():
                return "TRUE", 70.0, result
            elif "FALSE" in result.upper():
                return "FALSE", 70.0, result
            return "UNVERIFIABLE", 50.0, result
        except:
            return "UNVERIFIABLE", 50.0, "LLM failed"

    def check(self, text, true_label=None):
        start_time = time.time()
        original = text
        naive_tokens = self.count_tokens(original)

        # Preprocess
        processed = self.translate(text)
        compressed = self.compress_claim(processed)
        compressed_tokens = self.count_tokens(compressed)

        print("\n" + "="*70)
        print(f"📰 CLAIM: {original}")
        print(f"📝 Processed: {compressed}")
        print("-"*70)

        # *** WIKIPEDIA = HIGHEST PRIORITY ***
        wiki_verdict, wiki_conf, wiki_evidence, wiki_page = self.check_wikipedia_priority(compressed)

        final_verdict = "UNVERIFIABLE"
        confidence = 50.0
        explanation = ""
        evidence = ""

        # PRIORITY 1: Wikipedia Success = FINAL RESULT
        if wiki_verdict and wiki_conf > 75:
            final_verdict = wiki_verdict
            confidence = wiki_conf
            explanation = f"Wikipedia '{wiki_page}' confirms claim"
            evidence = wiki_evidence
            source = "Wikipedia Verified"
        else:
            # PRIORITY 2: Supporting evidence shown (but doesn't override wiki)
            print("\n🔧 SUPPORTING CHECKS:")
            ml_verdict, ml_conf = self.ml_check(compressed)
            faiss_evidence, faiss_conf = self.faiss_search(compressed)
            
            print(f"   ML: {ml_verdict} ({ml_conf:.1f}%)")
            if faiss_evidence:
                print(f"   FAISS: {faiss_conf:.1f}%")
            
            # PRIORITY 3: LLM Fallback (with wiki evidence)
            print("\n🤖 AI FALLBACK...")
            llm_verdict, llm_conf, llm_reason = self.llm_fallback(compressed, wiki_evidence)
            
            final_verdict = llm_verdict
            confidence = llm_conf
            explanation = llm_reason
            source = "AI Analysis"
            evidence = wiki_evidence or faiss_evidence or ""

        # Store evaluation
        if true_label in ["REAL", "FAKE"]:
            self.y_true.append("TRUE" if true_label == "REAL" else "FALSE")
            self.y_pred.append(final_verdict)

        # Results
        latency = time.time() - start_time
        self.display_result(original, final_verdict, confidence, source, explanation, evidence, 
                          naive_tokens, compressed_tokens, latency)
        return {
            "verdict": final_verdict, "confidence": confidence, 
            "source": source, "explanation": explanation, "evidence": evidence
        }

    def display_result(self, claim, verdict, confidence, source, explanation, evidence, 
                      naive_tokens, compressed_tokens, latency):
        status = "✅ VERIFIED TRUE" if verdict == "TRUE" else "❌ VERIFIED FALSE" if verdict == "FALSE" else "❓ UNVERIFIABLE"
        
        print("\n" + "="*70)
        print("🎯 FINAL RESULT")
        print("="*70)
        print(f"📄 CLAIM      : {claim[:80]}...")
        print("-"*70)
        print(f"✅ STATUS     : {status}")
        print(f"📊 CONFIDENCE : {confidence:.1f}%")
        print(f"🔍 SOURCE     : {source}")
        print("-"*70)
        if evidence:
            print(f"📜 EVIDENCE   : {evidence[:120]}...")
        print(f"💡 EXPLANATION: {explanation}")
        print("-"*70)
        print(f"⚡ PERF: {round((1-compressed_tokens/max(naive_tokens,1))*100,1)}% saved | {latency:.2f}s")
        print("="*70 + "\n")

    def evaluate(self):
        if not self.y_true:
            print("\n📊 No evaluation data yet.")
            return
        acc = accuracy_score(self.y_true, self.y_pred) * 100
        print(f"\n📊 METRICS: {len(self.y_true)} checks | Accuracy: {acc:.1f}%")

# ============================================
# MAIN RUNNER
# ============================================
if __name__ == "__main__":
    fc = FactChecker()
    
    print("\n" + "="*70)
    print("🤖 WIKIPEDIA-PRIORITY FACT CHECKER")
    print("📚 Wikipedia evidence gets highest priority")
    print("• Enter claim • 'eval' metrics • 'exit' quit")
    print("="*70)

    while True:
        try:
            claim = input("\n🔍 Enter claim: ").strip()
            if not claim or claim.lower() == 'exit':
                fc.evaluate()
                break
            if claim.lower() == 'eval':
                fc.evaluate()
                continue

            label = input("🎯 True label (REAL/FAKE/skip): ").strip().upper()
            if label not in ["REAL", "FAKE"]:
                label = None

            fc.check(claim, label)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
