import streamlit as st
import numpy as np
import sqlite3, hashlib
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from io import StringIO
import pandas as pd

# ================= CONFIG =================
MIN_DATA = 15
PATTERN_WINDOW = 5
DB_NAME = "users_ai.db"

st.set_page_config(page_title="AI Wingo Predictor", layout="centered")

# ================= DB =================
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cur = conn.cursor()

cur.execute("""CREATE TABLE IF NOT EXISTS history (
    username TEXT,
    result INTEGER
)""")

cur.execute("""DROP TABLE IF EXISTS reports""")
cur.execute("""CREATE TABLE reports (
    time TEXT,
    prediction TEXT,
    actual TEXT,
    confidence REAL,
    result TEXT
)""")
conn.commit()

# ================= MODEL =================
def build_lstm():
    m = Sequential([
        LSTM(32, input_shape=(10,1)),
        Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    return m

# ================= SESSION =================
if "seq" not in st.session_state:
    st.session_state.seq = []
if "model" not in st.session_state:
    st.session_state.model = build_lstm()
if "X" not in st.session_state:
    st.session_state.X = []
if "y" not in st.session_state:
    st.session_state.y = []

# ================= HELPERS =================
def lstm_prob():
    if len(st.session_state.seq) < 10:
        return 0.5
    seq = np.array(st.session_state.seq[-10:]).reshape(1,10,1)
    return float(st.session_state.model.predict(seq, verbose=0)[0][0])

def recent_pattern(seq):
    w = seq[-PATTERN_WINDOW:]
    b = w.count(1)
    s = w.count(0)
    if b >= 3:
        return "BIG", b/5
    if s >= 3:
        return "SMALL", s/5
    return None, 0

def recent_balance(seq):
    w = seq[-15:]
    if not w:
        return 0.5
    return 1 - abs(w.count(1) - w.count(0)) / len(w)

# ================= UI =================
st.title("🧠 AI Wingo Predictor")

st.metric("Total Data", len(st.session_state.seq))

prediction = None
confidence = 0

if len(st.session_state.seq) >= MIN_DATA:
    pat, pat_strength = recent_pattern(st.session_state.seq)

    if pat:
        lstm_p = lstm_prob()
        bal = recent_balance(st.session_state.seq)

        raw_conf = (
            0.5 * pat_strength +
            0.3 * lstm_p +
            0.2 * bal
        )

        confidence = int(60 + raw_conf * 40)  # 60–100%

        if confidence >= 60:
            prediction = pat
            st.success(f"🎯 Prediction: {prediction}")
            st.write(f"Confidence: {confidence}%")
        else:
            st.warning("⏳ WAIT (Low Confidence)")
    else:
        st.warning("⏳ WAIT (No clear pattern)")
else:
    st.info("⏳ Learning… need 15 data")

# ================= CONFIRM =================
st.subheader("Confirm & Learn")
actual = st.selectbox("Actual Result", ["BIG", "SMALL"])

if st.button("Confirm"):
    val = 1 if actual == "BIG" else 0
    st.session_state.seq.append(val)

    if len(st.session_state.seq) >= 10:
        st.session_state.X.append(st.session_state.seq[-10:])
        st.session_state.y.append(val)
        if len(st.session_state.X) % 5 == 0:
            X = np.array(st.session_state.X).reshape(-1,10,1)
            y = np.array(st.session_state.y)
            st.session_state.model.fit(X, y, epochs=2, verbose=0)

    if prediction:
        result = "WIN" if prediction == actual else "LOSS"
    else:
        result = "WAIT"

    cur.execute("INSERT INTO reports VALUES (?,?,?,?,?)",
                (str(datetime.now()), prediction, actual, confidence, result))
    conn.commit()

    st.success(f"Saved → {result}")

# ================= CSV =================
st.divider()
cur.execute("SELECT * FROM reports")
rows = cur.fetchall()

if rows:
    df = pd.DataFrame(rows, columns=["Time","Prediction","Actual","Confidence","Result"])
    buf = StringIO()
    df.to_csv(buf, index=False)
    st.download_button("⬇️ Download CSV", buf.getvalue(), "wingo_report.csv")
