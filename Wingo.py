import streamlit as st
import numpy as np
from collections import defaultdict, deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from io import StringIO
import pandas as pd

# ================= CONFIG =================
MIN_DATA = 15
SHORT_WIN = 5
LONG_WIN = 15
DB_MAX = 200

st.set_page_config(page_title="AI Wingo Predictor", layout="centered")

# ================= SESSION =================
if "seq" not in st.session_state:
    st.session_state.seq = []
if "markov" not in st.session_state:
    st.session_state.markov = defaultdict(lambda: defaultdict(int))
if "model" not in st.session_state:
    model = Sequential([
        LSTM(32, input_shape=(10,1)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    st.session_state.model = model
if "X" not in st.session_state:
    st.session_state.X, st.session_state.y = [], []
if "log" not in st.session_state:
    st.session_state.log = []
if "pending_actual" not in st.session_state:
    st.session_state.pending_actual = None

# ================= HELPERS =================
def pattern_score(seq, window):
    w = seq[-window:]
    b, s = w.count(1), w.count(0)
    if b > s:
        return "BIG", b / window
    if s > b:
        return "SMALL", s / window
    return None, 0

def markov_prob(seq):
    if len(seq) < 2:
        return 0.5
    prev = seq[-1]
    nxt = st.session_state.markov[prev]
    total = nxt[0] + nxt[1]
    if total == 0:
        return 0.5
    return nxt[1] / total

def lstm_prob(seq):
    if len(seq) < 10:
        return 0.5
    x = np.array(seq[-10:]).reshape(1,10,1)
    return float(st.session_state.model.predict(x, verbose=0)[0][0])

# ================= UI =================
st.title("🧠 AI Wingo Predictor (Markov + LSTM + Pattern AI)")
st.metric("Total Data", len(st.session_state.seq))

prediction, confidence = None, 0

if len(st.session_state.seq) >= MIN_DATA:
    sp, ss = pattern_score(st.session_state.seq, SHORT_WIN)
    lp, ls = pattern_score(st.session_state.seq, LONG_WIN)

    if sp and lp and sp == lp:
        mp = markov_prob(st.session_state.seq)
        lpb = lstm_prob(st.session_state.seq)

        final_score = (
            0.30 * ss +
            0.25 * ls +
            0.25 * mp +
            0.20 * lpb
        )

        confidence = int(60 + final_score * 40)

        if confidence >= 60 and mp >= 0.55:
            prediction = sp
            st.success(f"🎯 Prediction: {prediction}")
            st.write(f"Confidence: {confidence}%")
            st.write(f"Markov Next-State Prob: {mp:.2f}")
        else:
            st.warning("⏳ WAIT (Weak transition confidence)")
    else:
        st.warning("⏳ WAIT (Pattern disagreement)")
else:
    st.info("⏳ Learning… need 15 data")

# ================= CONFIRM & LEARN (BUG FIXED) =================
st.subheader("Confirm & Learn")

actual = st.radio("Actual Result (temporary)", ["BIG", "SMALL"])

if st.button("Confirm & Learn"):
    val = 1 if actual == "BIG" else 0

    # Markov update (AFTER confirmation only)
    if st.session_state.seq:
        prev = st.session_state.seq[-1]
        st.session_state.markov[prev][val] += 1

    st.session_state.seq.append(val)
    st.session_state.seq = st.session_state.seq[-DB_MAX:]

    # LSTM training
    if len(st.session_state.seq) >= 10:
        st.session_state.X.append(st.session_state.seq[-10:])
        st.session_state.y.append(val)
        if len(st.session_state.X) % 5 == 0:
            X = np.array(st.session_state.X).reshape(-1,10,1)
            y = np.array(st.session_state.y)
            st.session_state.model.fit(X, y, epochs=2, verbose=0)

    result = "WAIT"
    if prediction:
        result = "WIN" if prediction == actual else "LOSS"

    st.session_state.log.append({
        "Time": datetime.now(),
        "Prediction": prediction,
        "Actual": actual,
        "Confidence": confidence,
        "Result": result
    })

    st.success(f"Saved → {result}")

# ================= CSV =================
st.divider()
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    buf = StringIO()
    df.to_csv(buf, index=False)
    st.download_button("⬇️ Download CSV", buf.getvalue(), "wingo_ai_report.csv")
