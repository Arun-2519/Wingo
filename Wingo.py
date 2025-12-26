import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from io import BytesIO
import json
import os

# ================= CONFIG =================
MIN_DATA = 10
MIN_PATTERN_COUNT = 5
MIN_PATTERN_CONF = 0.65
POST_LOSS_WAIT = 1
DATA_FILE = "pattern_memory.json"

st.set_page_config(page_title="🔵🔴 BIG vs SMALL AI", layout="centered")
st.title("🔵 BIG vs 🔴 BIG–SMALL AI Predictor (Persistent & Stable)")

# ================= LOAD / SAVE MEMORY =================
def load_memory():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(mem):
    with open(DATA_FILE, "w") as f:
        json.dump(mem, f)

if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

if "inputs" not in st.session_state:
    st.session_state.inputs = []

if "log" not in st.session_state:
    st.session_state.log = []

if "post_loss_wait" not in st.session_state:
    st.session_state.post_loss_wait = 0

# ================= HELPERS =================
def get_patterns(seq):
    pats = []
    for k in range(3, 9):  # short + long
        if len(seq) >= k:
            pats.append(tuple(seq[-k:]))
    return pats

def update_memory(pattern, actual):
    mem = st.session_state.memory
    key = str(pattern)
    if key not in mem:
        mem[key] = {"BIG": [0,0], "SMALL": [0,0]}
    mem[key][actual][1] += 1

def update_win(pattern, prediction):
    key = str(pattern)
    st.session_state.memory[key][prediction][0] += 1

def best_pattern(seq):
    candidates = []
    for pat in get_patterns(seq):
        key = str(pat)
        stats = st.session_state.memory.get(key)
        if not stats:
            continue
        for res, (win, tot) in stats.items():
            if tot >= MIN_PATTERN_COUNT:
                rate = win / tot
                if rate >= MIN_PATTERN_CONF:
                    candidates.append((res, rate, len(pat)))
    if candidates:
        # Prefer longer patterns
        return max(candidates, key=lambda x: (x[1], x[2]))
    return None, 0, 0

# ================= PREDICTION =================
prediction, confidence = None, 0
history = st.session_state.inputs.copy()

if st.session_state.post_loss_wait > 0:
    st.warning("⏳ WAIT (post-loss stabilization)")

elif len(history) >= MIN_DATA:

    pred, conf, plen = best_pattern(history)

    if pred:
        prediction = pred
        confidence = int(conf * 100)
        st.success(f"🎯 Prediction: {prediction}")
        st.write(f"Confidence: {confidence}% | Pattern length: {plen}")
    else:
        st.warning("⏳ WAIT (no confirmed pattern)")
else:
    st.warning("🕐 Collecting data...")

# ================= INPUT =================
actual = st.selectbox("Enter Actual Result", ["BIG", "SMALL"])

if st.button("Confirm & Learn"):

    st.session_state.inputs.append(actual)

    # Update all pattern memories
    for pat in get_patterns(st.session_state.inputs[:-1]):
        update_memory(pat, actual)

        if prediction and prediction == actual:
            update_win(pat, prediction)

    result = "WAIT"
    if prediction:
        if prediction == actual:
            result = "WIN"
            st.session_state.post_loss_wait = 0
        else:
            result = "LOSS"
            st.session_state.post_loss_wait = POST_LOSS_WAIT

    save_memory(st.session_state.memory)

    st.session_state.log.append({
        "Prediction": prediction,
        "Actual": actual,
        "Confidence": confidence,
        "Result": result
    })

    st.success(f"Saved → {result}")
    st.rerun()

# ================= HISTORY =================
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)
    buf = BytesIO()
    df.to_excel(buf, index=False)
    st.download_button("⬇️ Download Excel", buf.getvalue(), "big_small_ai_persistent.xlsx")

st.caption("Persistent pattern AI • Short + Long memory • Multi-day learning")
