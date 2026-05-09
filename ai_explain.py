from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def explain_signal(ticker, signal, confidence, sentiment, rsi, macd):
    prompt = f"""
You are a financial analyst assistant. Given the following data for {ticker}, 
write a concise 3-sentence analysis for a retail investor.

Data:
- Signal: {signal} (model confidence: {confidence:.1%})
- News sentiment: {sentiment}
- RSI: {rsi:.1f}
- MACD: {macd:.2f}

Be factual and always end with: "This is not financial advice."
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    explanation = explain_signal(
        ticker="RELIANCE.NS",
        signal="BUY",
        confidence=0.63,
        sentiment="positive",
        rsi=59.0,
        macd=20.3
    )
    print(explanation)