import streamlit as st
import plotly.graph_objects as go
from data import get_stock_data, add_indicators
from model import train_model, get_signal
from sentiment import get_news_sentiment
from ai_explain import explain_signal

st.set_page_config(page_title="AI Stock Analyser", layout="wide")
st.title("AI Stock Analyser")
st.caption("For informational and educational purposes only. Not financial advice.")

WATCHLIST = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"]
ticker = st.sidebar.selectbox("Select stock", WATCHLIST)

with st.spinner("Fetching data..."):
    df = get_stock_data(ticker)
    df = add_indicators(df)
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

col1, col2, col3 = st.columns(3)
col1.metric("Current price", f"₹{df['Close'].iloc[-1]:,.2f}")
col2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
col3.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="Price"
))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50", line=dict(color="blue")))
fig.update_layout(title=f"{ticker} — Price chart", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Model signal")
with st.spinner("Running model..."):
    model, features = train_model(ticker)
    signal, confidence = get_signal(model, df, features)

signal_color = "green" if signal == "BUY" else "red"
st.markdown(f"### :{signal_color}[{signal}] — Confidence: {confidence:.1%}")

st.subheader("News sentiment")
with st.spinner("Analysing news..."):
    sentiment, sent_conf, headlines = get_news_sentiment(ticker)
st.write(f"Sentiment: **{sentiment}** ({sent_conf:.1%} confidence)")
for h in headlines:
    st.write(f"- {h}")

st.subheader("AI analysis")
with st.spinner("Generating explanation..."):
    explanation = explain_signal(
        ticker, signal, confidence, sentiment,
        df["RSI"].iloc[-1], df["MACD"].iloc[-1]
    )
st.info(explanation)