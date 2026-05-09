from transformers import pipeline
import requests

sentiment_model = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)

TICKER_NAMES = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "WIPRO.NS": "Wipro"
}

def get_sentiment(texts):
    if not texts:
        return "neutral", 0.0
    results = sentiment_model(texts[:5], truncation=True, max_length=512)
    scores = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        scores[r["label"].lower()] += r["score"]
    dominant = max(scores, key=scores.get)
    confidence = scores[dominant] / len(results)
    return dominant, round(confidence, 3)

def get_news_sentiment(ticker):
    company = TICKER_NAMES.get(ticker, ticker.replace(".NS", ""))
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company}&newsCount=5&region=IN"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        news = resp.json().get("news", [])
        headlines = [n["title"] for n in news if "title" in n]
        if not headlines:
            return "neutral", 0.0, []
        sentiment, confidence = get_sentiment(headlines)
        return sentiment, confidence, headlines
    except Exception as e:
        return "neutral", 0.0, []