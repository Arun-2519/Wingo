import streamlit as st
import pandas as pd
import numpy as np
from collections import deque, defaultdict
from io import BytesIO
from sklearn.naive_bayes import MultinomialNB
import math

# ================= CONFIG =================
MIN_DATA = 10
BASE_CONF = 65
MIN_CONF = 60
LOSS_LIMIT = 2
COOLDOWN_ROUNDS = 2
POST_LOSS_WAIT = 1

st.set_page_config(page_title="🔵🔴 BIG vs SMALL AI", layout="centered")
st.title("🔵 BIG vs 🔴 BIG–SMALL AI Predictor (REGIME AWARE)")

# ================= SESSION STATE =================
if "inputs" not in st.session_state:
    st.session_state.inputs = []

if "pattern_stats" not in st.session_state:
    st.session_state.pattern_stats = defaultdict(lambda: defaultdict(lambda: [0,0]))
    # pattern -> result -> [wins, total]

if "X_train" not in st.session_state:
    st.session_state.X_train = []

if "y_train" not in st.session_state:
    st.session_state.y_train = []

if "log" not in st.session_state:
    st.session_state.log = []

if "loss_streak" not in st.session_state:
    st.session_state.loss_streak = 0

if "cooldown" not in st.session_state:
    st.session_state.cooldown = 0

if "post_loss_wait" not in st.session_state:
    st.session_state.post_loss_wait = 0

# ================= HELPERS =================
ENC = {"SMALL": 0, "BIG": 1}
DEC = {0: "SMALL", 1: "BIG"}

def encode(seq):
    return [ENC[s] for s in seq]

def entropy(seq):
    if len(seq) == 0:
        return 1
    p = seq.count("BIG") / len(seq)
    if p in [0,1]:
        return 0
    return -p*math.log2(p)-(1-p)*math.log2(1-p)

def detect_chop(seq):
    if len(seq) < 6:
        return True
    return entropy(seq[-6:]) > 0.9

def get_recent_patterns(seq):
    pats = []
    for k in [3,4]:
        if len(seq) >= k:
            pats.append(tuple(seq[-k:]))
    return pats

# ================= PREDICTION =================
prediction, confidence = None, 0
history = st.session_state.inputs.copy()

if st.session_state.cooldown > 0:
    st.warning(f"⏳ COOLDOWN ({st.session_state.cooldown})")

elif st.session_state.post_loss_wait > 0:
    st.warning("⏳ WAIT (post-loss stabilization)")

elif len(history) >= MIN_DATA:

    if detect_chop(history):
        st.warning("⏳ WAIT (choppy / noisy regime)")
    else:
        candidates = []

        for pat in get_recent_patterns(history):
            stats = st.session_state.pattern_stats.get(pat, {})
            for res, (win, tot) in stats.items():
                if tot >= 3:
                    conf = win / tot
                    candidates.append((res, conf))

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            confidence = int(best[1] * 100)

            if confidence >= MIN_CONF:
                prediction = best[0]
                st.success(f"🎯 Prediction: {prediction}")
                st.write(f"Confidence: {confidence}%")
            else:
                st.warning("⏳ WAIT (weak evidence)")
        else:
            st.warning("⏳ WAIT (no stable pattern)")
else:
    st.warning("🕐 Collecting data...")

# ================= INPUT =================
actual = st.selectbox("Enter Actual Result", ["BIG","SMALL"])

if st.button("Confirm & Learn"):

    if st.session_state.cooldown > 0:
        st.session_state.cooldown -= 1

    if st.session_state.post_loss_wait > 0:
        st.session_state.post_loss_wait -= 1

    st.session_state.inputs.append(actual)

    # update pattern stats
    for pat in get_recent_patterns(st.session_state.inputs[:-1]):
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
    st.download_button("⬇️ Download Excel", buf.getvalue(), "big_small_ai_regime_v3.xlsx")

st.caption("Regime-aware AI • Evidence-based confidence • Loss-cluster control")
