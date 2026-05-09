import yfinance as yf
import pandas as pd
import ta
import streamlit as st

WATCHLIST = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"]

@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["MACD"] = ta.trend.MACD(df["Close"]).macd()
    df["SMA_20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["SMA_50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df

def get_all_stocks():
    all_data = {}
    for ticker in WATCHLIST:
        try:
            df = get_stock_data(ticker)
            df = add_indicators(df)
            all_data[ticker] = df
            print(f"{ticker}: ready with indicators")
        except Exception as e:
            print(f"{ticker}: failed — {e}")
    return all_data