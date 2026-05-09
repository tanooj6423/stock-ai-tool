import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data import get_stock_data, add_indicators

def build_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = ["RSI", "MACD", "SMA_20", "SMA_50", "BB_upper", "BB_lower", "Return"]
    df.dropna(inplace=True)
    return df[features], df["Target"]

def train_model(ticker="RELIANCE.NS"):
    df = get_stock_data(ticker, period="2y")
    df = add_indicators(df)
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{ticker} model accuracy: {acc:.1%}")
    return model, X.columns.tolist()

def get_signal(model, df, feature_cols):
    latest = df[feature_cols].dropna().iloc[-1:]
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][pred]
    signal = "BUY" if pred == 1 else "SELL"
    return signal, prob

if __name__ == "__main__":
    model, features = train_model("RELIANCE.NS")
    df = get_stock_data("RELIANCE.NS")
    df = add_indicators(df)
    df["Return"] = df["Close"].pct_change()
    signal, confidence = get_signal(model, df, features)
    print(f"Signal: {signal} | Confidence: {confidence:.1%}")