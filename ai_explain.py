from groq import Groq
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

def get_client():
    key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    return Groq(api_key=key)

def explain_signal(ticker, signal, confidence, sentiment,
                   rsi, macd, accuracy,
                   pe_ratio=None, week52_high=None,
                   week52_low=None, current_price=None):

    fundamentals = ""
    if pe_ratio and pe_ratio != "N/A":
        fundamentals += f"- P/E Ratio: {round(pe_ratio, 1)}\n"
    if week52_high and week52_low and week52_high != "N/A":
        fundamentals += f"- 52-week range: ₹{week52_low} — ₹{week52_high}\n"
        if current_price:
            pct = ((current_price - week52_high) / week52_high) * 100
            fundamentals += f"- Price vs 52W high: {pct:.1f}%\n"

    prompt = f"""You are a professional equity analyst specialising in Indian NSE stocks.
Analyse {ticker} and write a structured 4-point analysis for a retail investor.

Technical signals:
- Model signal: {signal} (confidence: {confidence:.1%}, backtested accuracy: {accuracy:.1%})
- RSI: {rsi:.1f} ({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral range'})
- MACD: {macd:.2f}
- News sentiment: {sentiment}
{f'Fundamentals:{chr(10)}{fundamentals}' if fundamentals else ''}

Write exactly 4 sentences:
1. What the technical signals indicate
2. What sentiment and momentum suggest
3. The key risk a trader should watch
4. A one-line summary verdict

End with: "This is not financial advice."
"""
    response = get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content