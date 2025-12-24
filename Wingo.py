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
MIN_START = 15
SHORT_PATTERN = 5          # last 4–5 pattern window
DOMINANCE_REQ = 3          # min dominance in window
BASE_THRESHOLD = 0.65
DB_NAME = "users_ai.db"

st.set_page_config(page_title="AI Wingo Predictor", layout="centered")

# ================= DATABASE =================
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS history (
    username TEXT,
    result INTEGER
)
""")

# Reset reports safely (development mode)
cur.execute("DROP TABLE IF EXISTS reports")
cur.execute("""
CREATE TABLE reports (
    username TEXT,
    time TEXT,
    prediction TEXT,
    actual TEXT,
    confidence REAL,
    pattern_window TEXT,
    result_status TEXT
)
""")
conn.commit()

# ================= UTILS =================
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ================= SOUND =================
SOUND_HIGH = "https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg"
SOUND_WAIT = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"

# ================= MODEL =================
def build_lstm():
    m = Sequential([
        LSTM(64, input_shape=(WINDOW,1), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    return m

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

# ================= APP =================
st.title("🧠 AI Wingo Predictor")

if not st.session_state.user:
    login_ui()
    st.stop()

if st.button("Logout"):
    st.session_state.clear()
    st.rerun()

# ================= INIT =================
if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.seq = []
    st.session_state.short = deque(maxlen=WINDOW)
    st.session_state.X, st.session_state.y = [], []
    st.session_state.model = build_lstm()
    st.session_state.pending_prediction = None

    cur.execute("SELECT result FROM history WHERE username=?", (st.session_state.user,))
    for (r,) in cur.fetchall():
        st.session_state.seq.append(r)
        st.session_state.short.append(r)

# ================= HELPERS =================
def detect_recent_pattern(seq):
    if len(seq) < SHORT_PATTERN:
        return None, 0

    window = seq[-SHORT_PATTERN:]
    ones = window.count(1)
    zeros = window.count(0)

    if ones >= DOMINANCE_REQ:
        return 1, ones
    if zeros >= DOMINANCE_REQ:
        return 0, zeros

    return None, 0

def lstm_confidence():
    if len(st.session_state.short) < WINDOW:
        return 0.5
    seq = np.array(st.session_state.short).reshape(1, WINDOW, 1)
    return float(st.session_state.model.predict(seq, verbose=0)[0][0])

# ================= UI =================
total = len(st.session_state.seq)
st.metric("Total Data Learned", total)

prediction = None
confidence = 0.0
pattern_value = None
pattern_count = 0

if total < MIN_START:
    st.info("⏳ Learning… waiting for 15 data")
    st.audio(SOUND_WAIT)
else:
    pattern_value, pattern_count = detect_recent_pattern(st.session_state.seq)

    if pattern_value is None:
        st.warning("⏳ WAIT FOR PATTERN (last 4–5 unstable)")
        st.audio(SOUND_WAIT)
    else:
        confidence = lstm_confidence()
        prediction = "BIG" if pattern_value == 1 else "SMALL"
        st.session_state.pending_prediction = prediction

        st.success(f"🎯 Prediction: {prediction}")
        st.write(f"Pattern strength: {pattern_count}/{SHORT_PATTERN}")
        st.write(f"LSTM confidence: {confidence*100:.2f}%")
        st.audio(SOUND_HIGH)

# ================= CONFIRM =================
st.subheader("Confirm & Learn")
actual = st.selectbox("Enter actual result", ["BIG", "SMALL"])

if st.button("Confirm & Learn"):
    actual_val = 1 if actual == "BIG" else 0

    # Save history
    cur.execute("INSERT INTO history VALUES (?,?)", (st.session_state.user, actual_val))
    conn.commit()

    st.session_state.seq.append(actual_val)
    st.session_state.short.append(actual_val)

    if len(st.session_state.short) == WINDOW:
        st.session_state.X.append(list(st.session_state.short))
        st.session_state.y.append(actual_val)
        if len(st.session_state.X) % 5 == 0:
            X = np.array(st.session_state.X).reshape(-1, WINDOW, 1)
            y = np.array(st.session_state.y)
            st.session_state.model.fit(X, y, epochs=2, verbose=0)

    if st.session_state.pending_prediction is None:
        result_status = "WAIT"
    else:
        result_status = "WIN" if st.session_state.pending_prediction == actual else "LOSS"

    cur.execute(
        "INSERT INTO reports VALUES (?,?,?,?,?,?,?)",
        (
            st.session_state.user,
            str(datetime.now()),
            st.session_state.pending_prediction,
            actual,
            confidence * 100,
            str(st.session_state.seq[-SHORT_PATTERN:]),
            result_status
        )
    )
    conn.commit()

    st.session_state.pending_prediction = None
    st.success(f"Saved ✔ Result: {result_status}")

# ================= CSV =================
st.divider()
st.subheader("📊 Download Report (CSV)")

cur.execute("""
SELECT time, prediction, actual, confidence, pattern_window, result_status
FROM reports WHERE username=?
""", (st.session_state.user,))
rows = cur.fetchall()

if rows:
    df = pd.DataFrame(rows, columns=[
        "Time","Prediction","Actual","Confidence","Last_4_5_Window","Result"
    ])
    buf = StringIO()
    df.to_csv(buf, index=False)

    st.download_button(
        "⬇️ Download CSV",
        data=buf.getvalue(),
        file_name="wingo_report.csv",
        mime="text/csv"
    )
else:
    st.info("No data yet.")
