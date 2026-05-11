---
title: Stock AI Tool
emoji: 📈
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.57.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-ML%20Engine-006400?style=for-the-badge"/>
<img src="https://img.shields.io/badge/NSE-Live%20Data-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/AI-Llama%203-8A2BE2?style=for-the-badge"/>

# NSE Stock Intelligence Platform

### Professional-grade AI equity analysis for Indian markets

</div>

---

## Overview

A full-stack quantitative research platform combining machine learning, natural language processing, and real-time market data — built for traders and investors who demand more than a chart.

---

## Core modules

| Module | Technology | Function |
|---|---|---|
| Signal engine | XGBoost + 15 features | Buy/sell signal with confidence score |
| Price forecast | Facebook Prophet | 30-day trend projection with confidence bands |
| Sentiment layer | finBERT NLP | Real-time news sentiment scoring |
| AI analyst | Llama 3 via Groq | Structured plain-English trade thesis |
| Screener | Multi-model voting | Full watchlist ranked by signal strength |
| Fundamentals | yfinance API | P/E, market cap, beta, 52-week range |

---

## Signal methodology
Live NSE price data (yfinance)
↓
Feature engineering — RSI, MACD, Bollinger Bands,
EMA, SMA cross, ATR, volume ratio, momentum, stochastic
↓
XGBoost classifier trained on 2 years of history
↓
Ensemble signal — BUY / SELL + confidence %
↓
finBERT news sentiment overlay
↓
Llama 3 trade thesis generation

---

## Technical indicators

`RSI` `MACD` `MACD Signal` `SMA 20` `SMA 50` `EMA 12` `Bollinger Bands` `ATR` `Stochastic` `Volume Ratio` `Rate of Change` `SMA Cross` `Price to SMA20` `Daily Return` `BB Width`

---

## Stack

| Layer | Tools |
|---|---|
| Data | yfinance · NSE API |
| Models | XGBoost · scikit-learn · Facebook Prophet |
| NLP | ProsusAI/finBERT · HuggingFace Transformers |
| AI | Llama 3.3 70B via Groq API |
| Frontend | Streamlit · Plotly |
| Hosting | HuggingFace Spaces |

---

## Disclaimer

> This platform is for **informational and educational purposes only**. Nothing presented here constitutes financial advice, investment recommendations, or solicitation to buy or sell securities. Past model performance does not guarantee future results. Always conduct independent research and consult a SEBI-registered advisor before making investment decisions.