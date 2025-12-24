import streamlit as st
import numpy as np
import sqlite3, hashlib
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
from datetime import datetime
from io import StringIO

# ================= CONFIG =================
WINDOW = 10
MIN_LEARN = 20
CONF_THRESHOLD = 0.65
STABILITY_THRESHOLD = 0.60
DB_NAME = "users_ai.db"

st.set_page_config(page_title="AI Wingo Predictor", layout="centered")

# ================= DATABASE =================
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cur = conn.cursor()

cur.execute("""CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT)""")

cur.execute("""CREATE TABLE IF NOT EXISTS history (
    username TEXT,
    result INTEGER)""")

cur.execute("""CREATE TABLE IF NOT EXISTS reports (
    username TEXT,
    time TEXT,
    confidence REAL,
    stability REAL,
    prediction TEXT,
    actual TEXT,
    note TEXT)""")

conn.commit()

# ================= UTILS =================
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ================= SOUND =================
SOUND_HIGH = "https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg"
SOUND_WARN = "https://actions.google.com/sounds/v1/alarms/warning.ogg"
SOUND_WAIT = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"

# ================= LSTM =================
def build_lstm():
    model = Sequential([
        LSTM(64, input_shape=(WINDOW,1), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    return model

# ================= LOGIN =================
if "user" not in st.session_state:
    st.session_state.user = None

def login_ui():
    st.subheader("🔐 Login / Signup")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    c1, c2 = st.columns(2)

    if c1.button("Login"):
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_pw(p)))
        if cur.fetchone():
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid credentials")

    if c2.button("Signup"):
        try:
            cur.execute("INSERT INTO users VALUES (?,?)", (u, hash_pw(p)))
            conn.commit()
            st.success("Account created. Login now.")
        except:
            st.error("Username exists")

# ================= APP START =================
st.title("🧠 AI Wingo Predictor")

if not st.session_state.user:
    login_ui()
    st.stop()

if st.button("Logout"):
    st.session_state.clear()
    st.rerun()

# ================= INIT USER MEMORY =================
if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.short = deque(maxlen=WINDOW)
    st.session_state.markov = {0:{0:1,1:1}, 1:{0:1,1:1}}
    st.session_state.global_c = {0:1,1:1}
    st.session_state.prev = None
    st.session_state.X, st.session_state.y = [], []
    st.session_state.model = build_lstm()
    st.session_state.pending_prediction = None

    cur.execute("SELECT result FROM history WHERE username=?", (st.session_state.user,))
    for (r,) in cur.fetchall():
        st.session_state.short.append(r)
        st.session_state.global_c[r] += 1
        if st.session_state.prev is not None:
            st.session_state.markov[st.session_state.prev][r] += 1
        st.session_state.prev = r

# ================= MODEL HELPERS =================
def lstm_prob():
    if len(st.session_state.X) < MIN_LEARN:
        return 0.5
    seq = np.array(st.session_state.short).reshape(1,WINDOW,1)
    return float(st.session_state.model.predict(seq, verbose=0)[0][0])

def ensemble_prob():
    m = st.session_state.markov[st.session_state.prev]
    m_prob = m[1] / (m[0]+m[1])
    g = st.session_state.global_c
    g_prob = g[1] / (g[0]+g[1])
    l_prob = lstm_prob()
    return 0.3*m_prob + 0.2*g_prob + 0.5*l_prob

def pattern_stability():
    m = st.session_state.markov[st.session_state.prev]
    total = m[0] + m[1]
    return max(m[0], m[1]) / total

# ================= UI =================
st.metric("Learned Rounds", len(st.session_state.X))

prediction = None
confidence = 0.0
stability = 0.0
note = "LEARNING"

if st.session_state.prev is not None:
    if len(st.session_state.X) < MIN_LEARN:
        st.info("⏳ WAIT FOR PATTERN (Learning...)")
        st.audio(SOUND_WAIT)
    else:
        confidence = ensemble_prob()
        stability = pattern_stability()

        if confidence >= CONF_THRESHOLD and stability >= STABILITY_THRESHOLD:
            prediction = "BIG" if confidence >= 0.5 else "SMALL"
            st.session_state.pending_prediction = prediction
            note = "READY"

            st.success(f"🎯 Prediction: {prediction}")
            st.write(f"Confidence: {confidence*100:.2f}%")
            st.write(f"Pattern Stability: {stability:.2f}")
            st.audio(SOUND_HIGH)
        else:
            note = "WAIT"
            st.warning("⚠️ WAIT FOR PATTERN")
            st.audio(SOUND_WARN)

# ================= CONFIRM & LEARN =================
st.subheader("Confirm & Learn")
actual = st.selectbox("Enter actual result", ["BIG", "SMALL"])

if st.button("Confirm & Learn"):
    actual_val = 1 if actual == "BIG" else 0

    cur.execute("INSERT INTO history VALUES (?,?)", (st.session_state.user, actual_val))
    conn.commit()

    st.session_state.short.append(actual_val)
    st.session_state.global_c[actual_val] += 1
    if st.session_state.prev is not None:
        st.session_state.markov[st.session_state.prev][actual_val] += 1

    if len(st.session_state.short) == WINDOW:
        st.session_state.X.append(list(st.session_state.short))
        st.session_state.y.append(actual_val)
        if len(st.session_state.X) % 10 == 0:
            X = np.array(st.session_state.X).reshape(-1,WINDOW,1)
            y = np.array(st.session_state.y)
            st.session_state.model.fit(X, y, epochs=3, verbose=0)

    cur.execute(
        "INSERT INTO reports VALUES (?,?,?,?,?,?,?)",
        (st.session_state.user, str(datetime.now()),
         confidence*100, stability,
         st.session_state.pending_prediction,
         actual, note)
    )
    conn.commit()

    st.session_state.prev = actual_val
    st.session_state.pending_prediction = None
    st.success("Saved & model learned")

# ================= CSV EXPORT (SAFE) =================
st.divider()
st.subheader("📊 Download Report (CSV)")

cur.execute("""
SELECT time, confidence, stability, prediction, actual, note
FROM reports WHERE username=?
""", (st.session_state.user,))
rows = cur.fetchall()

if rows:
    df = pd.DataFrame(rows, columns=[
        "Time","Confidence","Pattern Stability","Prediction","Actual","Note"
    ])
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)

    st.download_button(
        "⬇️ Download CSV",
        data=csv_buf.getvalue(),
        file_name="prediction_report.csv",
        mime="text/csv"
    )
else:
    st.info("No data yet.")
