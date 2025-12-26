import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from io import BytesIO
import json, os, math
from datetime import date

# ================= CONFIG =================
MIN_TOTAL_DATA = 10          # overall minimum
DAILY_WARMUP = 5             # new inputs required each day
MIN_CONF = 60
LOSS_LIMIT = 2
COOLDOWN_ROUNDS = 2
POST_LOSS_WAIT = 1

DATA_DIR = "ai_memory"
HISTORY_FILE = f"{DATA_DIR}/history.json"
PATTERN_FILE = f"{DATA_DIR}/pattern_stats.json"

os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="🔵🔴 BIG vs SMALL AI", layout="centered")
st.title("🔵 BIG vs 🔴 BIG–SMALL AI Predictor (LONG-TERM LEARNING)")

# ================= LOAD / SAVE =================
def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = load_json(HISTORY_FILE, [])

if "pattern_stats" not in st.session_state:
    raw = load_json(PATTERN_FILE, {})
    st.session_state.pattern_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for k, v in raw.items():
        st.session_state.pattern_stats[tuple(eval(k))] = v

if "today_date" not in st.session_state:
    st.session_state.today_date = str(date.today())

if "today_inputs" not in st.session_state:
    st.session_state.today_inputs = 0

if "loss_streak" not in st.session_state:
    st.session_state.loss_streak = 0

if "cooldown" not in st.session_state:
    st.session_state.cooldown = 0

if "post_loss_wait" not in st.session_state:
    st.session_state.post_loss_wait = 0

# reset daily counter
if st.session_state.today_date != str(date.today()):
    st.session_state.today_date = str(date.today())
    st.session_state.today_inputs = 0

# ================= HELPERS =================
def entropy(seq):
    if not seq:
        return 1
    p = seq.count("BIG") / len(seq)
    if p in (0, 1):
        return 0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def detect_chop(seq):
    if len(seq) < 6:
        return True
    return entropy(seq[-6:]) > 0.9

def extract_patterns(seq):
    patterns = []
    for k in range(3, 9):  # short + long patterns
        if len(seq) >= k:
            patterns.append(tuple(seq[-k:]))
    return patterns

# ================= PREDICTION =================
prediction = None
confidence = 0
history = st.session_state.history

if st.session_state.cooldown > 0:
    st.warning(f"⏳ COOLDOWN ({st.session_state.cooldown})")

elif st.session_state.post_loss_wait > 0:
    st.warning("⏳ WAIT (post-loss stabilization)")

elif len(history) >= MIN_TOTAL_DATA and st.session_state.today_inputs >= DAILY_WARMUP:

    if detect_chop(history):
        st.warning("⏳ WAIT (choppy regime)")
    else:
        candidates = []

        for pat in extract_patterns(history):
            stats = st.session_state.pattern_stats.get(pat, {})
            for res, (win, total) in stats.items():
                if total >= 3:
                    conf = win / total
                    candidates.append((res, conf))

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            confidence = int(best[1] * 100)

            if confidence >= MIN_CONF:
                prediction = best[0]
                st.success(f"🎯 Prediction: {prediction}")
                st.write(f"Confidence: {confidence}%")
            else:
                st.warning("⏳ WAIT (weak historical evidence)")
        else:
            st.warning("⏳ WAIT (no stable pattern yet)")
else:
    st.warning("🕐 Warming up with today’s data...")

# ================= INPUT =================
actual = st.selectbox("Enter Actual Result", ["BIG", "SMALL"])

if st.button("Confirm & Learn"):

    if st.session_state.cooldown > 0:
        st.session_state.cooldown -= 1

    if st.session_state.post_loss_wait > 0:
        st.session_state.post_loss_wait -= 1

    st.session_state.history.append(actual)
    st.session_state.today_inputs += 1

    # update pattern stats
    for pat in extract_patterns(st.session_state.history[:-1]):
        if prediction:
            st.session_state.pattern_stats[pat][prediction][1] += 1
            if prediction == actual:
                st.session_state.pattern_stats[pat][prediction][0] += 1

    result = "WAIT"
    if prediction:
        if prediction == actual:
            result = "WIN"
            st.session_state.loss_streak = 0
        else:
            result = "LOSS"
            st.session_state.loss_streak += 1
            st.session_state.post_loss_wait = POST_LOSS_WAIT
            if st.session_state.loss_streak >= LOSS_LIMIT:
                st.session_state.cooldown = COOLDOWN_ROUNDS

    save_json(HISTORY_FILE, st.session_state.history)
    save_json(PATTERN_FILE, {str(k): v for k, v in st.session_state.pattern_stats.items()})

    st.success(f"Saved → {result}")
    st.rerun()

# ================= HISTORY VIEW =================
if st.session_state.history:
    st.subheader("📊 Stored History (Persistent)")
    df = pd.DataFrame({"Result": st.session_state.history})
    st.dataframe(df.tail(50))

st.caption("Persistent AI • Short & Long Pattern Memory • Day-by-Day Learning")
