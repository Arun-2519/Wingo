import streamlit as st
import numpy as np
import sqlite3, hashlib
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import io, wave, math, pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
WINDOW = 10
MIN_LEARN = 10
MAX_LEARN = 20
DB_NAME = "users_ai.db"

st.set_page_config(page_title="AI Wingo Predictor", layout="centered")

# =========================
# DATABASE
# =========================
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

cur.execute("""
CREATE TABLE IF NOT EXISTS reports (
    username TEXT,
    time TEXT,
    prediction TEXT,
    confidence REAL,
    actual TEXT,
    outcome TEXT,
    loss_streak INTEGER
)
""")
conn.commit()

# =========================
# UTILS
# =========================
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# =========================
# SOUND
# =========================
def beep(freq=440, dur=0.25, rate=44100):
    t = np.linspace(0, dur, int(rate * dur), False)
    tone = np.sin(freq * t * 2 * math.pi)
    audio = np.int16(tone / np.max(np.abs(tone)) * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()

SOUND_HIGH = beep(900)
SOUND_WARN = beep(500)
SOUND_STOP = beep(250)

def play(sound):
    st.audio(sound, format="audio/wav")

# =========================
# LOGIN
# =========================
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
            st.success("Signup successful. Login now.")
        except:
            st.error("Username already exists")

# =========================
# MODEL
# =========================
def build_lstm():
    model = Sequential([
        LSTM(64, input_shape=(WINDOW, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    return model

# =========================
# APP START
# =========================
st.title("🧠 AI Wingo Predictor")

if not st.session_state.user:
    login_ui()
    st.stop()

st.success(f"Logged in as {st.session_state.user}")
if st.button("Logout"):
    st.session_state.clear()
    st.rerun()

# =========================
# INIT USER STATE
# =========================
if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.short = deque(maxlen=WINDOW)
    st.session_state.markov = {0:{0:1,1:1}, 1:{0:1,1:1}}
    st.session_state.global_c = {0:1,1:1}
    st.session_state.loss = 0
    st.session_state.win = 0
    st.session_state.prev = None
    st.session_state.X, st.session_state.y = [], []
    st.session_state.model = build_lstm()

    cur.execute("SELECT result FROM history WHERE username=?", (st.session_state.user,))
    for r in cur.fetchall():
        r = r[0]
        st.session_state.short.append(r)
        st.session_state.global_c[r] += 1
        if st.session_state.prev is not None:
            st.session_state.markov[st.session_state.prev][r] += 1
        st.session_state.prev = r

# =========================
# PREDICTION
# =========================
def lstm_prob():
    if len(st.session_state.X) < 20 or len(st.session_state.short) < WINDOW:
        return 0.5
    seq = np.array(st.session_state.short).reshape(1, WINDOW, 1)
    return float(st.session_state.model.predict(seq, verbose=0)[0][0])

def ensemble():
    m = st.session_state.markov[st.session_state.prev]
    m_prob = m[1] / (m[0] + m[1])
    g = st.session_state.global_c
    g_prob = g[1] / (g[0] + g[1])
    l_prob = lstm_prob()
    p = 0.3*m_prob + 0.2*g_prob + 0.5*l_prob
    return max(p - st.session_state.loss*0.05, 0)

# =========================
# DASHBOARD
# =========================
st.metric("Loss Streak", st.session_state.loss)
st.metric("Win Streak", st.session_state.win)
st.metric("Learned Rounds", len(st.session_state.X))

prediction = None
confidence = 0

if st.session_state.prev is not None:
    if len(st.session_state.X) < MIN_LEARN:
        st.info(f"Learning mode… Add {MIN_LEARN - len(st.session_state.X)} more results")
    else:
        confidence = ensemble()
        prediction = "BIG" if confidence >= 0.5 else "SMALL"

        st.subheader("Prediction")
        st.write("Result:", prediction)
        st.write("Confidence:", f"{confidence*100:.2f}%")

        if confidence < 0.5:
            st.error("🚫 DON'T BET (Low Confidence)")
        elif st.session_state.loss >= 4:
            play(SOUND_STOP)
            st.error("⛔ VERY HIGH RISK")
        elif st.session_state.loss >= 2:
            play(SOUND_WARN)
            st.warning("⚠️ CAUTION")
        elif confidence >= 0.7:
            play(SOUND_HIGH)

# =========================
# INPUT ACTUAL RESULT
# =========================
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
            X = np.array(st.session_state.X).reshape(-1, WINDOW, 1)
            y = np.array(st.session_state.y)
            st.session_state.model.fit(X, y, epochs=3, verbose=0)

    outcome = "LEARN"
    if prediction:
        if (prediction == "BIG" and actual == 1) or (prediction == "SMALL" and actual == 0):
            st.session_state.win += 1
            st.session_state.loss = 0
            outcome = "WIN"
        else:
            st.session_state.loss += 1
            st.session_state.win = 0
            outcome = "LOSS"

    cur.execute(
        "INSERT INTO reports VALUES (?,?,?,?,?,?,?)",
        (st.session_state.user, str(datetime.now()), prediction, confidence*100,
         "BIG" if actual==1 else "SMALL", outcome, st.session_state.loss)
    )
    conn.commit()

    st.session_state.prev = actual
    st.success("Saved & learned")

# =========================
# EXCEL EXPORT
# =========================
st.divider()
st.subheader("📊 Download Excel Report")

cur.execute("SELECT time,prediction,confidence,actual,outcome,loss_streak FROM reports WHERE username=?",
            (st.session_state.user,))
rows = cur.fetchall()

if rows:
    df = pd.DataFrame(rows, columns=[
        "Time", "Prediction", "Confidence", "Actual", "Outcome", "Loss Streak"
    ])
    st.download_button(
        "⬇️ Download Excel",
        df.to_excel(index=False),
        file_name="prediction_report.xlsx"
    )
else:
    st.info("No data to export yet.")
