# from flask import Flask, render_template, request
# import joblib
# import re, string

# # ======================
# # Load model and vectorizer
# # ======================
# model = joblib.load("models/logreg_model.joblib")
# vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# # ======================
# # Preprocessing (same as train.py)
# # ======================
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'\d+', '', text)  # remove numbers
#     text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
#     text = text.strip()
#     return text

# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     confidence = None
#     claim = ""

#     if request.method == "POST":
#         claim = request.form.get("claim")
#         if claim:
#             # Clean input the same way as training
#             cleaned = clean_text(claim)

#             # Vectorize
#             X_tfidf = vectorizer.transform([cleaned])

#             # Prediction
#             prob = model.predict_proba(X_tfidf)[0]
#             label = model.predict(X_tfidf)[0]

#             # ⚠️ Adjust this if dataset labels are reversed
#             # Check your dataset distribution in train.py:
#             # print(data['label'].value_counts())
#             prediction = "Real ✅" if label == 1 else "Fake ❌"
#             confidence = f"{max(prob) * 100:.2f}%"

#     return render_template(
#         "index.html",
#         prediction=prediction,
#         confidence=confidence,
#         claim=claim,
#     )

# if __name__ == "__main__":
#     app.run(debug=True)





# app.py
import os
import re
import string
import time
import html
from functools import lru_cache

import joblib
import requests
from flask import Flask, render_template, request

# OpenAI client (official new SDK usage pattern)
from openai import OpenAI

# --------------------------
# Configuration (set these env vars)
# --------------------------
OPENAI_API_KEY = os.environ.get("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")      # REQUIRED for GPT reasoning
NEWSAPI_KEY = os.environ.get("0fe031faf26f4e84a8aa792716e7e723")           # Recommended (news-only)
GNEWS_API_KEY = os.environ.get("e60ae33f022d3fd8d205755576817e89")       # Optional fallback
USE_NEWSAPI = bool(NEWSAPI_KEY)

# Limits
MAX_NEWS_RESULTS = 5

# --------------------------
# Load model & vectorizer (robust to different filenames)
# --------------------------
def load_model_and_vectorizer():
    model_paths = [
        "models/logreg_model.joblib",
        "models/logreg_model.pkl",
        "/mnt/data/logreg_model.pkl",
        "/mnt/data/logreg_model.joblib",
    ]
    vec_paths = [
        "models/tfidf_vectorizer.joblib",
        "models/tfidf_vectorizer.pkl",
        "/mnt/data/tfidf_vectorizer.pkl",
        "/mnt/data/tfidf_vectorizer.joblib",
    ]

    model = None
    vectorizer = None

    for p in model_paths:
        if os.path.exists(p):
            model = joblib.load(p)
            break
    for p in vec_paths:
        if os.path.exists(p):
            vectorizer = joblib.load(p)
            break

    if model is None or vectorizer is None:
        raise FileNotFoundError(
            "Could not find model/vectorizer. Put them in 'models/' as "
            "'logreg_model.joblib' and 'tfidf_vectorizer.joblib' (or "
            "upload to /mnt/data/ with those names)."
        )
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# --------------------------
# OpenAI client
# --------------------------
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# --------------------------
# Helpers: text cleaning
# --------------------------
def clean_text(text: str) -> str:
    text = str(text)
    text = text.strip()
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# --------------------------
# News retrieval (NewsAPI preferred, then GNews, then DuckDuckGo as last fallback)
# --------------------------
def get_newsapi_sources(query, max_results=MAX_NEWS_RESULTS, domains=None):
    if not NEWSAPI_KEY:
        return []
    base = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "apiKey": NEWSAPI_KEY,
    }
    if domains:
        # domains as comma-separated string
        params["domains"] = ",".join(domains)
    try:
        resp = requests.get(base, params=params, timeout=8)
        resp.raise_for_status()
        j = resp.json()
        articles = []
        for a in j.get("articles", [])[:max_results]:
            articles.append({
                "title": a.get("title"),
                "description": a.get("description") or "",
                "url": a.get("url"),
                "source": a.get("source", {}).get("name", ""),
                "publishedAt": a.get("publishedAt", "")
            })
        return articles
    except Exception:
        return []

def get_gnews_sources(query, max_results=MAX_NEWS_RESULTS):
    if not GNEWS_API_KEY:
        return []
    try:
        # gnews.io API
        base = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "token": GNEWS_API_KEY,
            "lang": "en",
            "max": max_results,
            "sortby": "publishedAt"
        }
        resp = requests.get(base, params=params, timeout=8)
        resp.raise_for_status()
        j = resp.json()
        articles = []
        for a in j.get("articles", [])[:max_results]:
            articles.append({
                "title": a.get("title"),
                "description": a.get("description") or "",
                "url": a.get("url"),
                "source": a.get("source", {}).get("name", ""),
                "publishedAt": a.get("publishedAt", "")
            })
        return articles
    except Exception:
        return []

def get_duckduckgo_sources(query, max_results=MAX_NEWS_RESULTS):
    # last resort: attempt duckduckgo-search if installed
    try:
        from duckduckgo_search import DDGS
    except Exception:
        return []
    links = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                links.append({
                    "title": r.get("title"),
                    "description": r.get("body") or "",
                    "url": r.get("href"),
                    "source": r.get("domain"),
                    "publishedAt": ""
                })
    except Exception:
        pass
    return links

def get_sources(query, max_results=MAX_NEWS_RESULTS):
    # Try NewsAPI first (trusted news outlets). If empty, try gnews, then duckduckgo.
    sources = []
    if NEWSAPI_KEY:
        sources = get_newsapi_sources(query, max_results=max_results)
    if not sources and GNEWS_API_KEY:
        sources = get_gnews_sources(query, max_results=max_results)
    if not sources:
        sources = get_duckduckgo_sources(query, max_results=max_results)
    return sources

# --------------------------
# Lightweight relevance scoring (simple token overlap)
# --------------------------
STOPWORDS = {
    "the","a","an","is","are","was","were","in","on","at","by","for","to","of",
    "and","or","that","this","it","with","from","as","be","has","have"
}

def score_relevance(claim, text):
    # very simple score: number of non-stopword overlaps
    claim_tokens = [t for t in re.findall(r"\w+", claim.lower()) if t not in STOPWORDS]
    text_tokens = [t for t in re.findall(r"\w+", (text or "").lower()) if t not in STOPWORDS]
    if not claim_tokens:
        return 0
    overlap = sum(1 for t in claim_tokens if t in text_tokens)
    return overlap / max(1, len(claim_tokens))

def shortlist_sources(claim, sources, top_k=3):
    scored = []
    for s in sources:
        text = (s.get("title","") or "") + " " + (s.get("description","") or "")
        sc = score_relevance(claim, text)
        scored.append((sc, s))
    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    return [s for (sc,s) in scored_sorted if sc>0][:top_k] or [s for (_,s) in scored_sorted][:min(top_k, len(scored_sorted))]

# --------------------------
# Fact-checking via GPT (final reasoning)
# --------------------------
def ask_gpt_factcheck(claim, style_label, sources):
    """
    Ask the LLM to compare the claim with the provided trusted sources.
    We request a JSON-like structured answer: verdict, explanation, supporting_urls
    verdict: CONFIRMED / REFUTED / NO_EVIDENCE
    """
    if client is None:
        return {
            "verdict": "NO_LLM",
            "explanation": "No OpenAI API key configured; cannot run LLM-based fact check.",
            "supporting_urls": []
        }

    # Build a compact context with top sources
    lines = []
    for s in sources:
        lines.append(f"- {s.get('source','')}: {s.get('title','')} ({s.get('url','')})")
        if s.get("description"):
            lines.append(f"  snippet: {s.get('description')}")
    sources_text = "\n".join(lines) if lines else "No sources available."

    prompt = f"""
You are a careful fact-checker. Use only the information from the "Sources" below to decide whether the "Claim" is supported.

Claim:
\"\"\"{claim}\"\"\"

Classifier (style-based) output:
\"\"\"{style_label}\"\"\"

Sources:
{sources_text}

Task (strict):
1) Decide one verdict from: CONFIRMED / REFUTED / NO_EVIDENCE.
   - CONFIRMED: at least one trusted source explicitly states or strongly supports the claim.
   - REFUTED: at least one trusted source explicitly contradicts the claim.
   - NO_EVIDENCE: none of the trusted sources confirm or refute the claim.
2) Provide a 1-3 sentence explanation (cite which source(s) you used).
3) Return a JSON object ONLY (no extra text) with fields:
   - verdict: one of CONFIRMED / REFUTED / NO_EVIDENCE
   - explanation: string
   - supporting_urls: array of urls (may be empty)
Example:
{{"verdict":"NO_EVIDENCE","explanation":"None of the sources mention the claimed death.","supporting_urls":[]}}
Now answer.
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content": prompt}],
            max_tokens=400
        )
        text = resp.choices[0].message.content.strip()

        # Try to parse JSON out of the text (robust small parser)
        import json, re
        json_text_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_text_match:
            json_text = json_text_match.group(0)
            parsed = json.loads(json_text)
            # normalize
            verdict = parsed.get("verdict", "NO_EVIDENCE")
            explanation = parsed.get("explanation", "")
            supporting_urls = parsed.get("supporting_urls", [])
            return {"verdict": verdict, "explanation": explanation, "supporting_urls": supporting_urls}
        else:
            # fallback: if not JSON, return a conservative NO_EVIDENCE with the raw reply
            return {"verdict": "NO_EVIDENCE", "explanation": text, "supporting_urls": []}
    except Exception as e:
        return {"verdict": "NO_LLM_ERROR", "explanation": f"LLM error: {str(e)}", "supporting_urls": []}

# --------------------------
# Simple in-memory cache to avoid repeated expensive calls while debugging
# --------------------------
CACHE = {}

def cached_get_sources(query):
    key = ("sources", query.strip().lower())
    # cache for 60 seconds in development
    entry = CACHE.get(key)
    if entry and time.time() - entry["t"] < 60:
        return entry["v"]
    v = get_sources(query)
    CACHE[key] = {"v": v, "t": time.time()}
    return v

def cached_factcheck(claim, style_label, sources_key):
    key = ("fact", claim.strip().lower(), str(sources_key))
    entry = CACHE.get(key)
    if entry and time.time() - entry["t"] < 60:
        return entry["v"]
    v = ask_gpt_factcheck(claim, style_label, sources_key)
    CACHE[key] = {"v": v, "t": time.time()}
    return v

# --------------------------
# Flask app
# --------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/", methods=["GET", "POST"])
def index():
    claim = ""
    classifier_label = None
    classifier_confidence = None
    style_suspicion = None

    evidence_verdict = None
    evidence_explanation = None
    evidence_supporting_urls = []
    sources = []

    if request.method == "POST":
        claim = request.form.get("claim", "").strip()
        if claim:
            cleaned = clean_text(claim)
            vect = vectorizer.transform([cleaned])
            # style classifier outputs: label & prob
            try:
                prob = model.predict_proba(vect)[0]
                label_val = model.predict(vect)[0]
                # Assume dataset uses 1 = real, 0 = fake. Adjust if needed.
                classifier_label = "Real (style)" if int(label_val) == 1 else "Fake-looking (style)"
                classifier_confidence = f"{max(prob)*100:.2f}%"
                style_suspicion = {"label": classifier_label, "confidence": classifier_confidence}
            except Exception:
                classifier_label = "Unknown (model error)"
                classifier_confidence = None
                style_suspicion = {"label": classifier_label, "confidence": classifier_confidence}

            # 1) Get trusted news sources (NewsAPI preferred)
            raw_sources = cached_get_sources(claim)

            # 2) Shortlist the most relevant sources to compare
            top_sources = shortlist_sources(claim, raw_sources, top_k=4)

            # 3) Ask GPT to compare claim vs top sources
            factcheck = cached_factcheck(claim, classifier_label, top_sources)

            evidence_verdict = factcheck.get("verdict")
            evidence_explanation = factcheck.get("explanation")
            evidence_supporting_urls = factcheck.get("supporting_urls", [])

            # Use the top_sources for UI display
            sources = top_sources

    return render_template(
        "index.html",
        claim=claim,
        prediction_style=style_suspicion,
        evidence_verdict=evidence_verdict,
        evidence_explanation=evidence_explanation,
        evidence_supporting_urls=evidence_supporting_urls,
        sources=sources,
    )

if __name__ == "__main__":
    app.run(debug=True)
