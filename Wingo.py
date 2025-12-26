import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from io import BytesIO
from sklearn.naive_bayes import MultinomialNB

# ================= CONFIG =================
MIN_DATA = 10
BASE_CONF = 65
LOSS_LIMIT = 2

st.set_page_config(page_title="🔵🔴 BIG vs SMALL AI", layout="centered")
st.title("🔵 BIG vs 🔴 BIG–SMALL AI Predictor (DATA-DRIVEN FIXED)")

st.markdown("""
<style>
body { background-color: #0f1117; color: #ffffff; }
.stButton>button { background-color: #2196f3; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "inputs" not in st.session_state:
    st.session_state.inputs = []

if "X_train" not in st.session_state:
    st.session_state.X_train = []

if "y_train" not in st.session_state:
    st.session_state.y_train = []

if "log" not in st.session_state:
    st.session_state.log = []

if "loss_streak" not in st.session_state:
    st.session_state.loss_streak = 0

if "model_perf" not in st.session_state:
    st.session_state.model_perf = deque(maxlen=10)

# ================= HELPERS =================
ENC = {"SMALL": 0, "BIG": 1}
DEC = {0: "SMALL", 1: "BIG"}

def encode(seq):
    return [ENC[s] for s in seq]

def auto_threshold():
    if len(st.session_state.model_perf) < 5:
        return BASE_CONF
    acc = sum(st.session_state.model_perf) / len(st.session_state.model_perf)
    return int(np.clip(BASE_CONF + (0.55 - acc) * 25, 65, 80))

def current_streak(seq):
    if not seq:
        return None, 0
    last = seq[-1]
    length = 1
    for i in range(len(seq)-2, -1, -1):
        if seq[i] == last:
            length += 1
        else:
            break
    return last, length

def alternation_signal(seq):
    if len(seq) < 3:
        return None, 0
    a, b, c = seq[-3:]
    if a == c and a != b:
        # strong alternation
        return b, 0.75
    return None, 0

# ================= PREDICTION (PAST ONLY) =================
prediction, confidence = None, 0
conf_threshold = auto_threshold()

history = st.session_state.inputs.copy()

if len(history) >= MIN_DATA and st.session_state.loss_streak < LOSS_LIMIT:

    signals = []

    # ---------- Alternation (PRIMARY) ----------
    alt_pred, alt_conf = alternation_signal(history)
    if alt_pred:
        signals.append((alt_pred, alt_conf))

    # ---------- Streak BREAK logic ----------
    last, streak_len = current_streak(history)
    if streak_len >= 3:
        opp = "SMALL" if last == "BIG" else "BIG"
        signals.append((opp, 0.6))

    # ---------- Naive Bayes (SUPPORT ONLY) ----------
    if len(st.session_state.X_train) >= 10:
        clf = MultinomialNB()
        clf.fit(st.session_state.X_train, st.session_state.y_train)
        probs = clf.predict_proba([encode(history[-10:])])[0]
        idx = np.argmax(probs)
        nb_pred = DEC[idx]
        nb_conf = min(0.1, probs[idx] * 0.1)  # cap NB influence
        signals.append((nb_pred, nb_conf))

    if signals:
        preds = [s[0] for s in signals]
        strengths = [s[1] for s in signals]

        prediction = max(set(preds), key=preds.count)
        confidence = int(60 + sum(strengths) * 40)

        # penalty for long streaks
        if streak_len >= 4:
            confidence -= 20

        if confidence >= conf_threshold:
            st.success(f"🎯 Prediction: {prediction}")
            st.write(f"Confidence: {confidence}% (threshold {conf_threshold}%)")
        else:
            prediction = None
            st.warning("⏳ WAIT (low confidence)")
    else:
        st.warning("⏳ WAIT (no valid pattern)")
else:
    st.warning("🔒 PROFIT PROTECTION ACTIVE")

# ================= INPUT UI =================
st.subheader("🎮 Enter Actual Result")
actual = st.selectbox("Select result (temporary)", ["BIG", "SMALL"])

if st.button("Confirm & Learn"):
    st.session_state.inputs.append(actual)

    # train NB on past window only
    if len(st.session_state.inputs) >= 11:
        st.session_state.X_train.append(
            encode(st.session_state.inputs[-11:-1])
        )
        st.session_state.y_train.append(ENC[actual])

    result = "WAIT"
    if prediction:
        result = "WIN" if prediction == actual else "LOSS"
        st.session_state.loss_streak = 0 if result == "WIN" else st.session_state.loss_streak + 1
        st.session_state.model_perf.append(1 if result == "WIN" else 0)

    st.session_state.log.append({
        "Prediction": prediction,
        "Confidence": confidence,
        "Actual": actual,
        "Result": result
    })

    st.success(f"Saved → {result}")
    st.rerun()

# ================= HISTORY =================
if st.session_state.log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)

    buf = BytesIO()
    df.to_excel(buf, index=False)
    st.download_button("⬇️ Download Excel", buf.getvalue(), "big_small_ai_fixed_v2.xlsx")

st.markdown("---")
st.caption("Data-driven model: Alternation + Streak-Break + NB Support")
