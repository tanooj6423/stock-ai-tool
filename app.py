import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from data import get_stock_data, add_indicators, get_fundamentals, WATCHLIST
from model import train_model, get_signal
from sentiment import get_news_sentiment
from ai_explain import explain_signal

st.set_page_config(page_title="AI Stock Analyser", layout="wide", page_icon="📈")
st.title("📈 AI Stock Analyser")
st.caption("For informational and educational purposes only. Not financial advice.")

ticker = st.sidebar.selectbox("Select stock", WATCHLIST)
tab1, tab2, tab3 = st.tabs(["Analysis", "Price Forecast", "Screener"])

with tab1:
    with st.spinner("Fetching data..."):
        df = get_stock_data(ticker, period="2y")
        df = add_indicators(df)
        df["Return"] = df["Close"].pct_change()
        df["SMA_cross"] = (df["SMA_20"] - df["SMA_50"]) / df["SMA_50"]
        df["Price_to_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
        df.dropna(inplace=True)
        fundamentals = get_fundamentals(ticker)

    price = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]
    change = ((price - prev) / prev) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"₹{price:,.2f}", f"{change:+.2f}%")
    c2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    c3.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
    c4.metric("52W High", f"₹{fundamentals.get('52W High', 'N/A')}")
    c5.metric("52W Low", f"₹{fundamentals.get('52W Low', 'N/A')}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50", line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
                             line=dict(color="gray", width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
                             line=dict(color="gray", width=1, dash="dash"),
                             fill="tonexty", fillcolor="rgba(128,128,128,0.1)"))
    fig.update_layout(title=f"{ticker} — Price chart", xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="purple")))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(title="RSI", height=200, showlegend=False)
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.subheader("Model signal")
    with st.spinner("Running model..."):
        model, features, accuracy = train_model(ticker)
        signal, confidence = get_signal(model, df, features)
    color = "green" if signal == "BUY" else "red"
    col1, col2 = st.columns(2)
    col1.markdown(f"### :{color}[{signal}] — Confidence: {confidence:.1%}")
    col2.metric("Model accuracy", f"{accuracy:.1%}")

    if fundamentals:
        st.subheader("Fundamentals")
        fcols = st.columns(len(fundamentals))
        for i, (k, v) in enumerate(fundamentals.items()):
            if k == "Market Cap" and isinstance(v, (int, float)):
                fcols[i].metric(k, f"₹{v/1e9:.0f}B")
            elif k == "Dividend Yield" and isinstance(v, float):
                fcols[i].metric(k, f"{v:.1%}")
            else:
                fcols[i].metric(k, f"{round(v,2)}" if isinstance(v, float) else str(v))

    st.subheader("News sentiment")
    with st.spinner("Analysing news..."):
        sentiment, sent_conf, headlines = get_news_sentiment(ticker)
    color_s = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "gray"
    st.markdown(f"Sentiment: **:{color_s}[{sentiment}]** ({sent_conf:.1%} confidence)")
    for h in headlines:
        st.write(f"- {h}")

    st.subheader("AI analysis")
    with st.spinner("Generating explanation..."):
        explanation = explain_signal(
            ticker, signal, confidence, sentiment,
            df["RSI"].iloc[-1], df["MACD"].iloc[-1], accuracy,
            pe_ratio=fundamentals.get("PE Ratio"),
            week52_high=fundamentals.get("52W High"),
            week52_low=fundamentals.get("52W Low"),
            current_price=price
        )
    st.info(explanation)

with tab2:
    st.subheader("30-day price forecast")
    try:
        from prophet import Prophet
        ticker_f = st.selectbox("Select stock", WATCHLIST, key="fc")
        with st.spinner("Running forecast..."):
            df_f = get_stock_data(ticker_f, period="2y")
            pf = df_f.reset_index()[["Date", "Close"]].copy()
            pf.columns = ["ds", "y"]
            pf["ds"] = pf["ds"].dt.tz_localize(None)
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
            m.fit(pf)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=pf["ds"], y=pf["y"], name="Actual", line=dict(color="white")))
        fig_f.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="orange")))
        fig_f.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"],
                                   line=dict(color="rgba(255,165,0,0.2)"), showlegend=False))
        fig_f.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"],
                                   line=dict(color="rgba(255,165,0,0.2)"),
                                   fill="tonexty", fillcolor="rgba(255,165,0,0.1)", showlegend=False))
        fig_f.update_layout(title=f"{ticker_f} — 30-day forecast", height=500)
        st.plotly_chart(fig_f, use_container_width=True)
        last_forecast = forecast["yhat"].iloc[-1]
        last_actual = pf["y"].iloc[-1]
        change_f = ((last_forecast - last_actual) / last_actual) * 100
        st.metric("Forecast in 30 days", f"₹{last_forecast:,.2f}", f"{change_f:+.1f}%")
        st.caption("Based on trend and seasonality only. Not financial advice.")
    except ImportError:
        st.warning("Run `pip install prophet` to enable this tab.")

with tab3:
    st.subheader("Screener — all stocks ranked by signal")
    if st.button("Run screener"):
        results = []
        bar = st.progress(0)
        for i, t in enumerate(WATCHLIST):
            try:
                df_s = get_stock_data(t, period="2y")
                df_s = add_indicators(df_s)
                df_s["Return"] = df_s["Close"].pct_change()
                df_s["SMA_cross"] = (df_s["SMA_20"] - df_s["SMA_50"]) / df_s["SMA_50"]
                df_s["Price_to_SMA20"] = (df_s["Close"] - df_s["SMA_20"]) / df_s["SMA_20"]
                df_s.dropna(inplace=True)
                m_s, f_s, acc_s = train_model(t)
                sig_s, conf_s = get_signal(m_s, df_s, f_s)
                results.append({
                    "Ticker": t,
                    "Price": f"₹{df_s['Close'].iloc[-1]:,.2f}",
                    "Signal": sig_s,
                    "Confidence": f"{conf_s:.1%}",
                    "RSI": f"{df_s['RSI'].iloc[-1]:.1f}",
                    "Accuracy": f"{acc_s:.1%}"
                })
            except:
                results.append({"Ticker": t, "Signal": "Error"})
            bar.progress((i + 1) / len(WATCHLIST))
        st.dataframe(pd.DataFrame(results), use_container_width=True)