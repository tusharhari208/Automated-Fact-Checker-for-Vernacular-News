import os
import re
import sys
import pickle
import wikipedia
import google.generativeai as genai
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor
import scaledown as sd
from dotenv import load_dotenv
import os
import scaledown as sd

load_dotenv()
# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.verification.similarity_search import SimilaritySearchEngine
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")


class FactCheckerPipeline:
    def compress_claim(self, text):
        try:
            result = self.compressor.compress(
            context="",
            prompt="Extract the core factual claim in minimal words: " + text
        )
            return result.prompt   # compressed claim
        except:
            return text  #fallback
  
    
    def __init__(self, gemini_api_key):
        print("Loading Fact Checker Pipeline...")
        sd.set_api_key(os.getenv("SCALEDOWN_API_KEY"))

        self.compressor = ScaleDownCompressor(
        target_model="gpt-4o",
        rate="auto"
        )
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Load classical models (with fallback)
        try:
            with open(os.path.join(BASE_DIR, "saved_models", "tfidf.pkl"), "rb") as f:
                self.tfidf = pickle.load(f)

            with open(os.path.join(BASE_DIR, "saved_models", "passive_aggressive.pkl"), "rb") as f:
                self.pa_model = pickle.load(f)

            with open(os.path.join(BASE_DIR, "saved_models", "logistic_regression.pkl"), "rb") as f:
                self.lr_model = pickle.load(f)

            self.use_ml = True
            print("ML models loaded!")

        except:
            self.use_ml = False
            print("ML models not found → using FAISS-only mode")

        # Load similarity search engine
        self.search_engine = SimilaritySearchEngine()

        # Load Gemini LLM
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel("gemini-1.5-flash")
        print("Gemini LLM loaded!")

        print("Pipeline ready!")

    # ============================================
    # WIKIPEDIA SEARCH
    # ============================================
    def check_wikipedia(self, news_text):
        try:
            keywords     = " ".join(news_text.split()[:10])
            wiki_results = wikipedia.search(keywords, results=3)

            if not wiki_results:
                return {
                    "wiki_verdict"   : "NOT FOUND",
                    "wiki_summary"   : "No Wikipedia article found",
                    "wiki_confidence": 0
                }

            wiki_page = wikipedia.summary(
                wiki_results[0],
                sentences = 5
            )

            news_words   = set(news_text.lower().split())
            wiki_words   = set(wiki_page.lower().split())
            common_words = news_words.intersection(wiki_words)

            stop_words = {
                "the", "a", "an", "is", "are", "was",
                "were", "in", "on", "at", "to", "for",
                "of", "and", "or", "but", "it", "this",
                "that", "with", "have", "has", "had"
            }
            common_words = common_words - stop_words
            similarity   = len(common_words) / max(len(news_words), 1)

            if similarity > 0.3:
                wiki_verdict    = "SUPPORTED"
                wiki_confidence = round(similarity * 100, 2)
            elif similarity > 0.15:
                wiki_verdict    = "PARTIAL"
                wiki_confidence = round(similarity * 100, 2)
            else:
                wiki_verdict    = "NOT SUPPORTED"
                wiki_confidence = round(similarity * 100, 2)

            return {
                "wiki_verdict"   : wiki_verdict,
                "wiki_summary"   : wiki_page[:300] + "...",
                "wiki_confidence": wiki_confidence
            }

        except wikipedia.exceptions.DisambiguationError as e:
            try:
                wiki_page = wikipedia.summary(e.options[0], sentences=5)
                return {
                    "wiki_verdict"   : "PARTIAL",
                    "wiki_summary"   : wiki_page[:300] + "...",
                    "wiki_confidence": 30
                }
            except:
                return {
                    "wiki_verdict"   : "NOT FOUND",
                    "wiki_summary"   : "Disambiguation error",
                    "wiki_confidence": 0
                }

        except Exception as e:
            return {
                "wiki_verdict"   : "ERROR",
                "wiki_summary"   : "Wikipedia search failed: " + str(e),
                "wiki_confidence": 0
            }

    # ============================================
    # COMBINED VERDICT SCORE
    # ============================================
    def get_final_verdict(self, pa_result, lr_result,
                          evidence_verdict, wiki_verdict):
        score = 0

        score += 1 if pa_result       == "REAL"          else -1
        score += 1 if lr_result       == "REAL"          else -1

        if evidence_verdict   == "REAL":   score += 1
        elif evidence_verdict == "FAKE":   score -= 1

        if wiki_verdict       == "SUPPORTED":     score += 1
        elif wiki_verdict     == "NOT SUPPORTED": score -= 1

        if score > 0:   return "REAL"
        elif score < 0: return "FAKE"
        else:           return "UNVERIFIED"

    # ============================================
    # RAG + LLM EXPLANATION
    # ============================================
    def get_llm_explanation(self, news_text, pa_result,
                             lr_result, evidence_verdict,
                             wiki_summary, final_verdict,
                             top_evidence):
        """
        RAG FLOW:
        1. RETRIEVE  → Collect all evidence
                       (FAISS + Wikipedia results)
        2. AUGMENT   → Build a prompt with
                       all evidence as context
        3. GENERATE  → Gemini LLM reads context
                       and generates explanation
        """

        # STEP 1: RETRIEVE — collect all evidence
        evidence_text = ""
        for i, ev in enumerate(top_evidence[:2], 1):
            evidence_text += (
                "Evidence " + str(i) + ":\n" +
                "  Claim : " + ev["claim"] + "\n" +
                "  Label : " + ev["label"] + "\n\n"
            )

        # STEP 2: AUGMENT — build prompt with context
        prompt = """
You are a fact checking assistant.
Analyze the following news and evidence, then give a clear explanation.

NEWS TO CHECK:
{news}

MACHINE LEARNING MODELS VERDICT:
- Passive Aggressive Model : {pa}
- Logistic Regression Model: {lr}

SIMILAR VERIFIED CLAIMS FROM FEVER DATASET:
{evidence}

WIKIPEDIA SUMMARY:
{wiki}

FINAL VERDICT: {verdict}

Based on ALL the above evidence, explain in 3-4 sentences:
1. Why this news is {verdict}
2. What evidence supports this verdict
3. What the user should know about this news

Keep explanation simple, clear and helpful.
""".format(
            news     = news_text,
            pa       = pa_result,
            lr       = lr_result,
            evidence = evidence_text,
            wiki     = wiki_summary,
            verdict  = final_verdict
        )

        # STEP 3: GENERATE — Gemini generates explanation
        try:
            compressed_context = (evidence_text + " " + wiki_summary)[:300]

            compressed = self.compressor.compress(
                context=compressed_context,
                prompt=prompt
            )

            final_prompt = compressed.prompt

            response = self.llm.generate_content(final_prompt)
            explanation = response.text

        except Exception as e:
            explanation = "Could not generate explanation: " + str(e)

        return explanation

    # ============================================
    # MAIN CHECK FUNCTION
    # ============================================
    def check_news(self, news_text, use_wiki=False, use_llm=True):
        news_text = news_text[:300]
        compressed_news = self.compress_claim(news_text)
        compressed_claim = compressed_news
        print("\nCHECKING NEWS:")
        print(news_text[:100] + "...")
        print("=" * 50)

        # STAGE 1: CLASSICAL MODELS
        if self.use_ml:
            compressed_input = compressed_news
            text_tfidf = self.tfidf.transform([compressed_input])

            pa_pred = self.pa_model.predict(text_tfidf)[0]
            lr_pred = self.lr_model.predict(text_tfidf)[0]

            pa_result = "REAL" if pa_pred == 1 else "FAKE"
            lr_result = "REAL" if lr_pred == 1 else "FAKE"

            print("Stage 1 - Classical Models:")
            print("  Passive Aggressive : " + pa_result)
            print("  Logistic Regression: " + lr_result)
        else:
            pa_result = "SKIPPED"
            lr_result = "SKIPPED"
            print("Stage 1 - Skipped (no ML models)")
        
        # STAGE 2: SIMILARITY SEARCH
        print("\nStage 2 - Similarity Search...")
        compressed_claim = compressed_news
        verdict = self.search_engine.get_verdict(compressed_claim)
        print("  Evidence Verdict   : " + verdict["verdict"])
        print("  Confidence         : " + str(verdict["confidence"]) + "%")

        # STAGE 3: WIKIPEDIA CHECK
        print("\nStage 3 - Wikipedia Check...")

        if verdict["confidence"] < 70:
            print("  Low confidence → Using Wikipedia")
            compressed_query = compressed_news
            wiki = self.check_wikipedia(compressed_query)
            print("  Wikipedia Verdict  : " + wiki["wiki_verdict"])
            print("  Wiki Confidence    : " + str(wiki["wiki_confidence"]) + "%")

        else:
            wiki = {"wiki_verdict": "SKIPPED", "wiki_summary": "", "wiki_confidence": 0}
            print("  Wikipedia skipped (high FAISS confidence)")

        # STAGE 4: COMBINED VERDICT
        if self.use_ml:
            final_verdict = self.get_final_verdict(
                pa_result,
                lr_result,
                verdict["verdict"],
                wiki["wiki_verdict"]
                )
        else:
            if verdict["confidence"] > 75:
                    final_verdict = verdict["verdict"]
            elif wiki["wiki_verdict"] == "SUPPORTED":
                final_verdict = "REAL"
            elif wiki["wiki_verdict"] == "NOT SUPPORTED":
                final_verdict = "FAKE"
            else:
                final_verdict = "UNVERIFIED"
            print("\nStage 4 - Final Verdict: " + final_verdict)

        # STAGE 5: RAG + LLM EXPLANATION
        print("\nStage 5 - Generating LLM Explanation...")

        if final_verdict == "UNVERIFIED" or verdict["confidence"] < 60:
            print("  Using LLM (low confidence case)")

            explanation = self.get_llm_explanation(
            news_text,
            pa_result,
            lr_result,
            verdict["verdict"],        
            wiki["wiki_summary"],
            final_verdict,
            verdict["top_evidence"]    
)
        else:
            explanation = "LLM skipped for speed (high confidence result)"

        print("\n" + "=" * 50)
        print("FINAL VERDICT    : " + final_verdict)
        print("LLM EXPLANATION  : ")
        print(explanation)
        print("=" * 50)

        return {
            "pa_result"        : pa_result,
            "lr_result"        : lr_result,
            "evidence_verdict" : verdict["verdict"],
            "evidence_confidence": verdict["confidence"],
            "wiki_verdict"     : wiki["wiki_verdict"],
            "wiki_confidence"  : wiki["wiki_confidence"],
            "wiki_summary"     : wiki["wiki_summary"],
            "final_verdict"    : final_verdict,
            "explanation"      : explanation
        }

    # ============================================
    # PARSE NUMBERED INPUT
    # ============================================
    def parse_numbered_input(self, raw_input):
        lines     = raw_input.strip().split("\n")
        news_list = []
        current   = ""

        for line in lines:
            if re.match(r"^\d+\.", line.strip()):
                if current.strip():
                    news_list.append(current.strip())
                current = re.sub(r"^\d+\.\s*", "", line.strip())
            else:
                current += " " + line.strip()

        if current.strip():
            news_list.append(current.strip())

        return news_list


# ============================================
# MAIN WITH LOOP
# ============================================
if __name__ == "__main__":
    pipeline = FactCheckerPipeline(GEMINI_API_KEY)

    print("\n" + "=" * 50)
    print("      FAKE NEWS DETECTOR + LLM")
    print("FORMAT:")
    print("  1. News article one here")
    print("  2. News article two here")
    print("Press ENTER TWICE to submit.")
    print("Type 'quit' to exit.")
    print("=" * 50)

    while True:

        print("\nPaste your numbered news below:")
        lines = []

        while True:
            line = input()
            if line.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                exit()
            if line == "":
                break
            lines.append(line)

        raw_input = "\n".join(lines)

        if not raw_input.strip():
            print("Nothing entered! Please try again.")
            continue

        news_list = pipeline.parse_numbered_input(raw_input)

        if len(news_list) == 0:
            print("Could not find numbered news!")
            print("Use format: 1. news   2. news")
            continue

        print("\nFound " + str(len(news_list)) + " articles to check!")

        all_results = []
        for i, news in enumerate(news_list, 1):

            if len(news) < 10:
                print("\nArticle " + str(i) + " too short, skipping...")
                continue

            print("\n>>> ARTICLE " + str(i) + " of " + str(len(news_list)))
            result             = pipeline.check_news(news)
            result["news_text"] = news
            all_results.append(result)

        # FINAL SUMMARY
        print("\n" + "=" * 50)
        print("         SUMMARY OF ALL RESULTS")
        print("=" * 50)

        for i, result in enumerate(all_results, 1):
            print("\nArticle " + str(i) + ": " + result["news_text"][:60] + "...")
            print("  PA Model     : " + result["pa_result"])
            print("  LR Model     : " + result["lr_result"])
            print("  Evidence     : " + result["evidence_verdict"])
            print("  Wikipedia    : " + result["wiki_verdict"])
            print("  FINAL        : " + result["final_verdict"])
            print("  EXPLANATION  : ")
            print(result["explanation"])

        print("\n" + "=" * 50)
        print("Check more news? Paste again or type 'quit'")
        print("=" * 50)
if __name__ == "__main__":
        pipeline = FactCheckerPipeline(GEMINI_API_KEY)