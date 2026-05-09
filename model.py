import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from data import get_stock_data, add_indicators

FEATURE_COLS = [
    "RSI", "MACD", "MACD_signal", "SMA_20", "SMA_50", "EMA_12",
    "BB_upper", "BB_lower", "BB_width", "ATR",
    "Volume_ratio", "ROC", "Stoch",
    "Return", "SMA_cross", "Price_to_SMA20"
]

def build_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["SMA_cross"] = (df["SMA_20"] - df["SMA_50"]) / df["SMA_50"]
    df["Price_to_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    available = [f for f in FEATURE_COLS if f in df.columns]
    return df[available], df["Target"], available

@st.cache_resource
def train_model(ticker):
    df = get_stock_data(ticker, period="2y")
    df = add_indicators(df)
    X, y, features = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, features, round(acc, 3)

def get_signal(model, df, feature_cols):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["SMA_cross"] = (df["SMA_20"] - df["SMA_50"]) / df["SMA_50"]
    df["Price_to_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    available = [f for f in feature_cols if f in df.columns]
    latest = df[available].dropna().iloc[-1:]
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][pred]
    return ("BUY" if pred == 1 else "SELL"), prob