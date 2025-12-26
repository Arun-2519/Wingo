import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from io import BytesIO

from sklearn.naive_bayes import MultinomialNB

# ================= CONFIG =================
MIN_DATA = 10
BASE_CONF = 65
LOSS_LIMIT = 3

st.set_page_config(page_title="🔵🔴 BIG vs SMALL AI", layout="centered")
st.title("🔵 BIG vs 🔴 SMALL Predictor (AI Powered)")

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

# ordered pattern memory (3–5)
if "patterns" not in st.session_state:
    st.session_state.patterns = {
        k: defaultdict(lambda: defaultdict(int)) for k in range(3, 6)
    }

# performance tracking (auto-disable logic)
if "model_perf" not in st.session_state:
    st.session_state.model_perf = deque(maxlen=10)

# ================= HELPERS =================
ENC = {"SMALL": 0, "BIG": 1}
DEC = {0: "SMALL", 1: "BIG"}

def encode(seq):
    return [ENC[s] for s in seq]

def decode(v):
    return DEC.get(v, "")

def auto_threshold():
    if len(st.session_state.model_perf) < 5:
        return BASE_CONF
    acc = sum(st.session_state.model_perf) / len(st.session_state.model_perf)
    return int(np.clip(BASE_CONF + (0.6 - acc) * 20, 60, 75))

def regime_shift(seq):
    if len(seq) < 8:
        return False
    last = seq[-8:]
    # unstable if nearly equal BIG/SMALL
    return abs(last.count("BIG") - last.count("SMALL")) <= 1

def learn_patterns(seq):
    for k in range(3, 6):
        if len(seq) > k:
            key = tuple(seq[-(k+1):-1])
            nxt = seq[-1]
            st.session_state.patterns[k][key][nxt] += 1

def pattern_signal(seq):
    for k in range(5, 2, -1):
        key = tuple(seq[-k:])
        if key in st.session_state.patterns[k]:
            counts = st.session_state.patterns[k][key]
            total = sum(counts.values())
            if total >= 3:
                best = max(counts, key=counts.get)
                return best, counts[best] / total
    return None, 0

# ================= PREDICTION =================
prediction, confidence = None, 0
conf_threshold = auto_threshold()
regime = regime_shift(st.session_state.inputs)

if len(st.session_state.inputs) >= MIN_DATA and st.session_state.loss_streak < LOSS_LIMIT:

    signals = []

    # ---- Pattern AI ----
    p_pred, p_strength = pattern_signal(st.session_state.inputs)
    if p_pred:
        signals.append((p_pred, p_strength))

    # ---- Naive Bayes ----
    if len(st.session_state.X_train) >= 10:
        clf = MultinomialNB()
        clf.fit(st.session_state.X_train, st.session_state.y_train)
        encoded = encode(st.session_state.inputs[-10:])
        probs = clf.predict_proba([encoded])[0]
        idx = np.argmax(probs)
        signals.append((decode(idx), probs[idx]))

    if signals:
        preds = [s[0] for s in signals]
        strengths = [s[1] for s in signals]

        top = max(set(preds), key=preds.count)
        confidence = int(60 + np.mean(strengths) * 40)

        if regime:
            confidence -= 10

        if confidence >= conf_threshold:
            prediction = top
            st.success(f"🎯 Prediction: {prediction}")
            st.write(f"Confidence: {confidence}% (threshold {conf_threshold}%)")
        else:
            st.warning("⏳ WAIT (low confidence)")
    else:
        st.warning("⏳ WAIT (learning patterns)")
else:
    st.warning("🔒 PROFIT PROTECTION ACTIVE")

# ================= INPUT UI =================
st.subheader("🎮 Add Actual Result")
actual = st.selectbox("Select actual result (temporary)", ["BIG", "SMALL"])

if st.button("Confirm & Learn"):
    # ---- Learn ONLY here ----
    st.session_state.inputs.append(actual)

    if len(st.session_state.inputs) >= 10:
        st.session_state.X_train.append(
            encode(st.session_state.inputs[-10:])
        )
        st.session_state.y_train.append(ENC[actual])

    learn_patterns(st.session_state.inputs)

    result = "WAIT"
    if prediction:
        result = "WIN" if prediction == actual else "LOSS"
        st.session_state.loss_streak = (
            0 if result == "WIN" else st.session_state.loss_streak + 1
        )
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
    st.download_button(
        "⬇️ Download Excel",
        buf.getvalue(),
        "big_small_ai.xlsx"
    )

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Naive Bayes, Pattern AI & Self-Regulation")
