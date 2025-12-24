import streamlit as st
import numpy as np
import sqlite3, hashlib
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
from datetime import datetime
from io import BytesIO

# ================= CONFIG =================
WINDOW = 10
MIN_LEARN = 20
CONF_THRESHOLD = 0.65
DB_NAME = "users_ai.db"

st.set_page_config(page_title="AI Wingo Predictor", layout="centered")

# ================= DATABASE =================
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cur = conn.cursor()

# USERS
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

# HISTORY
cur.execute("""
CREATE TABLE IF NOT EXISTS history (
    username TEXT,
    result INTEGER
)
""")

# REPORTS (DROP & RECREATE SAFELY)
cur.execute("DROP TABLE IF EXISTS reports")
cur.execute("""
CREATE TABLE reports (
    username TEXT,
    time TEXT,
    confidence REAL,
    prediction TEXT,
    actual TEXT,
    note TEXT
)
""")
conn.commit()

# ================= UTILS =================
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

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
            st.error("Username already exists")

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

    cur.execute("SELECT result FROM history WHERE username=?", (st.session_state.user,))
    for (r,) in cur.fetchall():
        st.session_state.short.append(r)
        st.session_state.global_c[r] += 1
        if st.session_state.prev is not None:
            st.session_state.markov[st.session_state.prev][r] += 1
        st.session_state.prev = r

# ================= PREDICTION =================
def lstm_prob():
    if len(st.session_state.X) < MIN_LEARN:
        return 0.5
    seq = np.array(st.session_state.short).reshape(1,WINDOW,1)
    return float(st.session_state.model.predict(seq, verbose=0)[0][0])

def ensemble():
    m = st.session_state.markov[st.session_state.prev]
    m_prob = m[1] / (m[0]+m[1])
    g = st.session_state.global_c
    g_prob = g[1] / (g[0]+g[1])
    l_prob = lstm_prob()
    return 0.3*m_prob + 0.2*g_prob + 0.5*l_prob

# ================= UI =================
st.metric("Total Learned Rounds", len(st.session_state.X))

prediction = None
confidence = 0.0
note = "LEARNING"

if st.session_state.prev is not None:
    if len(st.session_state.X) < MIN_LEARN:
        st.info("⏳ WAIT FOR PATTERN (Learning...)")
    else:
        confidence = ensemble()
        if confidence >= CONF_THRESHOLD:
            prediction = "BIG" if confidence >= 0.5 else "SMALL"
            note = "PREDICTION"
            st.success(f"🎯 Prediction: {prediction}")
            st.write(f"Confidence: {confidence*100:.2f}%")
        else:
            note = "LOW_CONFIDENCE"
            st.warning("⏳ WAIT FOR PATTERN")

# ================= INPUT =================
st.subheader("Enter Actual Result")
c1, c2 = st.columns(2)
actual = None
if c1.button("BIG"): actual = 1
if c2.button("SMALL"): actual = 0

if actual is not None:
    cur.execute("INSERT INTO history VALUES (?,?)", (st.session_state.user, actual))
    conn.commit()

    st.session_state.short.append(actual)
    st.session_state.global_c[actual] += 1
    if st.session_state.prev is not None:
        st.session_state.markov[st.session_state.prev][actual] += 1

    if len(st.session_state.short) == WINDOW:
        st.session_state.X.append(list(st.session_state.short))
        st.session_state.y.append(actual)
        if len(st.session_state.X) % 10 == 0:
            X = np.array(st.session_state.X).reshape(-1,WINDOW,1)
            y = np.array(st.session_state.y)
            st.session_state.model.fit(X, y, epochs=3, verbose=0)

    cur.execute(
        "INSERT INTO reports VALUES (?,?,?,?,?,?)",
        (st.session_state.user, str(datetime.now()),
         confidence*100, prediction,
         "BIG" if actual==1 else "SMALL",
         note)
    )
    conn.commit()

    st.session_state.prev = actual
    st.success("Saved & learning continues")

# ================= EXCEL EXPORT =================
st.divider()
st.subheader("📊 Download Excel Report")

cur.execute("""
SELECT time, confidence, prediction, actual, note
FROM reports WHERE username=?
""", (st.session_state.user,))
rows = cur.fetchall()

if rows:
    df = pd.DataFrame(rows, columns=["Time","Confidence","Prediction","Actual","Note"])
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="xlsxwriter")
    buffer.seek(0)

    st.download_button(
        "⬇️ Download Excel",
        data=buffer,
        file_name="prediction_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("No report data yet.")
