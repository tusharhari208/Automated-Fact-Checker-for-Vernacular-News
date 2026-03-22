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
# FACT CHECKER PIPELINE - BEST FEATURES COMBINED
# ============================================
class FactChecker:
    def __init__(self):
        print("\n" + "="*70)
        print(" LOADING ADVANCED FACT CHECKER SYSTEM...")
        print("="*70)

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ML Models (Optional)
        try:
            with open(os.path.join(BASE_DIR, "saved_models", "tfidf.pkl"), "rb") as f:
                self.tfidf = pickle.load(f)
            with open(os.path.join(BASE_DIR, "saved_models", "logistic_regression.pkl"), "rb") as f:
                self.lr_model = pickle.load(f)
            self.use_ml = True
            print("ML models loaded!")
        except Exception as e:
            self.use_ml = False
            logging.warning(f"ML models not found: {e}")
            print("  ⚠️  ML models not found")

        # FAISS search engine
        self.search_engine = SimilaritySearchEngine()
        print(" FAISS loaded!")

        # NLI model (FIXED zero-shot pipeline)
        print("  🔄 Loading NLI model...")
        self.nli_model = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU
        )
        print("  ✅ NLI loaded!")

        # Gemini LLM
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

        # Evaluation storage
        self.y_true = []
        self.y_pred = []
        self.results_history = []

        print("="*70)
        print("  🎯 SYSTEM READY!")
        print("="*70)

    def translate(self, text):
        """STEP 1: Translate to English if needed"""
        try:
            if detect(text) != "en":
                text = GoogleTranslator(source='auto', target='en').translate(text)
            return text.lower().strip()
        except:
            return text.lower().strip()

    def compress_claim(self, text, max_len=150):
        """STEP 2: Compress using ScaleDown or truncate"""
        if not self.compressor:
            return text[:max_len]
        try:
            result = self.compressor.compress(
                context="", 
                prompt="Extract core factual claim briefly: " + text
            )
            return result.prompt[:max_len]
        except:
            return text[:max_len]

    def count_tokens(self, text):
        """Simple token estimation"""
        return self.token_count(text)

    def ml_check(self, text):
        """STEP 3: ML Classification (if available)"""
        if not self.use_ml:
            return "UNKNOWN", 0.0
        
        try:
            ml_features = self.tfidf.transform([text])
            ml_pred = self.lr_model.predict(ml_features)[0]
            ml_prob = self.lr_model.predict_proba(ml_features)[0]
            ml_confidence = max(ml_prob) * 100
            ml_verdict = "TRUE" if ml_pred == 1 else "FALSE"
            return ml_verdict, ml_confidence
        except Exception as e:
            logging.error(f"ML check failed: {e}")
            return "UNKNOWN", 0.0

    def faiss_search(self, text, top_k=5):
        """STEP 4: FAISS Similarity Search"""
        try:
            results = self.search_engine.search_evidence(text, top_k=top_k)
            strong = [r for r in results if r["score"] > 0.75]
            if strong:
                top = strong[0]
                return top["claim"], round(top["score"] * 100, 2)
            return None, 0.0
        except Exception as e:
            logging.error(f"FAISS search failed: {e}")
            return None, 0.0

    def run_nli(self, claim, evidence):
        """STEP 5: NLI Verification"""
        if not evidence:
            return "UNVERIFIABLE", 0.0
        
        try:
            result = self.nli_model(
                sequences=claim,
                candidate_labels=["entailment", "contradiction", "neutral"],
                hypothesis_template="This claim {} the following evidence: {}.".format("", evidence)
            )
            top_label = result["labels"][0]
            top_score = result["scores"][0] * 100
            
            if top_label == "entailment":
                return "TRUE", round(top_score, 2)
            elif top_label == "contradiction":
                return "FALSE", round(top_score, 2)
            else:
                return "UNVERIFIABLE", round(top_score, 2)
        except Exception as e:
            logging.error(f"NLI failed: {e}")
            return "UNVERIFIABLE", 0.0

    def check_wiki(self, text):
        """STEP 6: Wikipedia Fallback"""
        try:
            res = wikipedia.search(text, results=1)
            if not res:
                return {"found": False, "summary": ""}
            summary = wikipedia.summary(res[0], sentences=2)
            return {"found": True, "summary": self.compress_claim(summary)}
        except:
            return {"found": False, "summary": ""}

    def gemini_reason(self, claim, evidence_text):
        """STEP 7: Gemini Final Fallback"""
        if not self.llm:
            return "VERDICT: UNVERIFIABLE\nREASON: LLM unavailable"
        
        prompt = f"""You are an expert fact checker.
Claim: {claim}
Evidence: {evidence_text}

Respond EXACTLY in this format:
VERDICT: TRUE / FALSE / UNVERIFIABLE
REASON: [one clear sentence]"""

        try:
            if self.compressor:
                compressed = self.compressor.compress(context=evidence_text, prompt=prompt)
                final_prompt = compressed.prompt
            else:
                final_prompt = prompt
            
            response = self.llm.generate_content(final_prompt)
            return response.text.strip()
        except Exception as e:
            logging.error(f"Gemini failed: {e}")
            return "VERDICT: UNVERIFIABLE\nREASON: LLM processing failed"

    def check(self, text, true_label=None):
        """MAIN PIPELINE: All steps combined"""
        start_time = time.time()
        original = text
        naive_tokens = self.count_tokens(original)

        # STEP 1-2: Translate + Compress
        text = self.translate(text)
        compressed = self.compress_claim(text)
        compressed_tokens = self.count_tokens(compressed)

        print("\n" + "="*70)
        print(f"📰 CLAIM: {original}")
        print(f"📝 Compressed: {compressed}")
        print(f"⚡ Token Reduction: {round((1-compressed_tokens/max(naive_tokens,1))*100,1)}%")
        print("-"*70)

        # STEP 3: ML Check
        print("  1️⃣ ML CHECK:")
        ml_verdict, ml_conf = self.ml_check(compressed)
        print(f"     ML: {ml_verdict} ({ml_conf:.1f}%)")
        ml_norm = ml_verdict if ml_verdict != "UNKNOWN" else None

        # STEP 4: FAISS Search
        print("\n  2️⃣ FAISS SEARCH:")
        evidence, faiss_score = self.faiss_search(compressed)
        if evidence:
            print(f"     Score: {faiss_score:.1f}%")
            print(f"     Evidence: {evidence[:80]}...")
        else:
            print("     No strong matches")

        # STEP 5: NLI on FAISS evidence
        print("\n  3️⃣ NLI CHECK:")
        nli_verdict, nli_conf = self.run_nli(compressed, evidence)
        print(f"     NLI: {nli_verdict} ({nli_conf:.1f}%)")

        # DECISION LOGIC (Prioritized)
        final_verdict = "UNVERIFIABLE"
        confidence = 0.0
        source = "none"
        explanation = ""

        # CASE 1: ML + NLI Agreement (Highest confidence)
        if ml_norm and ml_norm == nli_verdict and nli_verdict != "UNVERIFIABLE" and nli_conf > 70:
            final_verdict = nli_verdict
            confidence = (ml_conf + nli_conf) / 2
            source = "ML + NLI Agreement"
            explanation = "Both ML classifier and NLI model strongly agree"

        # CASE 2: Strong NLI alone
        elif nli_verdict != "UNVERIFIABLE" and nli_conf > 85:
            final_verdict = nli_verdict
            confidence = nli_conf
            source = "Strong NLI"
            explanation = "NLI model highly confident"

        # CASE 3: Strong ML alone
        elif ml_norm and ml_conf > 85:
            final_verdict = ml_norm
            confidence = ml_conf
            source = "Strong ML"
            explanation = "ML classifier highly confident"

        # STEP 6: Wikipedia fallback
        if final_verdict == "UNVERIFIABLE":
            print("\n  4️⃣ WIKIPEDIA CHECK:")
            wiki = self.check_wiki(compressed)
            if wiki["found"]:
                print(f"     Wiki: {wiki['summary'][:80]}...")
                wiki_nli, wiki_conf = self.run_nli(compressed, wiki["summary"])
                print(f"     Wiki NLI: {wiki_nli} ({wiki_conf:.1f}%)")
                
                if wiki_nli != "UNVERIFIABLE" and wiki_conf > 70:
                    final_verdict = wiki_nli
                    confidence = wiki_conf
                    source = "Wikipedia + NLI"
                    explanation = "Wikipedia evidence verified by NLI"
                    evidence = wiki["summary"]

        # STEP 7: Gemini final fallback
        if final_verdict == "UNVERIFIABLE":
            print("\n  5️⃣ GEMINI FALLBACK:")
            evidence_text = (evidence or "") + (wiki["summary"] if wiki["found"] else "")
            gemini_result = self.gemini_reason(compressed, evidence_text)
            print(f"     Gemini: {gemini_result[:100]}")
            
            if "TRUE" in gemini_result.upper():
                final_verdict = "TRUE"
            elif "FALSE" in gemini_result.upper():
                final_verdict = "FALSE"
            source = "Gemini LLM"
            explanation = gemini_result

        # Store evaluation data
        if true_label in ["REAL", "FAKE"]:
            self.y_true.append("TRUE" if true_label == "REAL" else "FALSE")
            self.y_pred.append(final_verdict)
            self.results_history.append({
                "claim": original, "verdict": final_verdict, 
                "confidence": confidence, "source": source
            })

        # Performance metrics
        latency = time.time() - start_time
        
        result = self._build_result(final_verdict, confidence, source, explanation, evidence)
        self.display_result(original, result, naive_tokens, compressed_tokens, latency)
        
        return result

    def _build_result(self, verdict, confidence, source, explanation, evidence):
        return {
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "source": source,
            "explanation": explanation,
            "evidence": evidence[:150] if evidence else ""
        }

    def display_result(self, claim, result, naive_tokens, compressed_tokens, latency):
        """Enhanced result display"""
        verdict, confidence, source, explanation, evidence = (
            result["verdict"], result["confidence"], result["source"],
            result["explanation"], result["evidence"]
        )

        status = "✅ VERIFIED TRUE" if verdict == "TRUE" else "❌ VERIFIED FALSE" if verdict == "FALSE" else "❓ UNVERIFIABLE"
        
        print("\n" + "="*70)
        print("🎯 FINAL FACT CHECK RESULT")
        print("="*70)
        print(f"📄 CLAIM      : {claim[:80]}...")
        print("-"*70)
        print(f"✅ STATUS     : {status}")
        print(f"📊 CONFIDENCE : {confidence:.1f}%")
        print(f"🔍 SOURCE     : {source}")
        print("-"*70)
        if evidence:
            print(f"📜 EVIDENCE   : {evidence}")
        print(f"💡 EXPLANATION: {explanation[:120]}")
        print("-"*70)
        print(f"⚡ PERFORMANCE")
        print(f"   Tokens saved: {round((1-compressed_tokens/max(naive_tokens,1))*100,1)}%")
        print(f"   Latency    : {latency:.2f}s")
        print("="*70 + "\n")

    def evaluate(self):
        """Show evaluation metrics"""
        if not self.y_true:
            print("\n📊 No evaluation data. Provide true labels (REAL/FAKE) during checks.")
            return

        acc = accuracy_score(self.y_true, self.y_pred) * 100
        prec = precision_score(self.y_true, self.y_pred, average='macro', zero_division=0) * 100
        rec = recall_score(self.y_true, self.y_pred, average='macro', zero_division=0) * 100
        f1 = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0) * 100

        print("\n" + "="*70)
        print("📊 EVALUATION METRICS")
        print("="*70)
        print(f"Total predictions : {len(self.y_true)}")
        print(f"✅ Accuracy       : {acc:.2f}%")
        print(f"📈 Precision      : {prec:.2f}%")
        print(f"📉 Recall         : {rec:.2f}%")
        print(f"🎯 F1 Score       : {f1:.2f}%")
        print("="*70)

# ============================================
# MAIN RUNNER
# ============================================
if __name__ == "__main__":
    fc = FactChecker()
    
    print("\n" + "="*70)
    print("🤖 ADVANCED FACT CHECKER - MULTI-LAYER VERIFICATION")
    print("💎 Features: ML + FAISS + NLI + Wikipedia + Gemini")
    print("="*70)
    print("Commands:")
    print("• Enter claim to verify")
    print("• Type 'eval' for metrics")
    print("• Type 'exit' to quit")
    print("="*70)

    while True:
        try:
            claim = input("\n🔍 Enter claim: ").strip()
            
            if not claim:
                print("❌ Please enter a valid claim!")
                continue
                
            if claim.lower() == 'exit':
                print("\n👋 Thank you for using Advanced Fact Checker!")
                fc.evaluate()
                break
                
            if claim.lower() == 'eval':
                fc.evaluate()
                continue

            # Optional ground truth for evaluation
            label = input("🎯 True label (REAL/FAKE/Enter to skip): ").strip().upper()
            if label not in ["REAL", "FAKE"]:
                label = None

            fc.check(claim, label)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            fc.evaluate()
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print(f"❌ Error: {e}")