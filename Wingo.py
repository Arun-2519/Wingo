import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict, deque, Counter
from datetime import datetime
from io import StringIO

from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ================= CONFIG =================
MIN_DATA = 10
BASE_CONF = 65
LOSS_LIMIT = 2
SEQ_MAX = 5

st.set_page_config(page_title="🧠 Heavy AI Predictor", layout="wide")

# ================= SESSION INIT =================
def init_state():
    st.session_state.seq = []
    st.session_state.log = []
    st.session_state.loss_streak = 0

    st.session_state.model_perf = {
        "A": deque(maxlen=10),
        "B": deque(maxlen=10)
    }

    st.session_state.patterns = {k: defaultdict(lambda: defaultdict(int)) for k in range(3,6)}
    st.session_state.X_nb, st.session_state.y_nb = [], []

if "seq" not in st.session_state:
    init_state()

# ================= TRANSFORMER =================
def build_transformer():
    inp = tf.keras.Input(shape=(SEQ_MAX,))
    x = Embedding(2, 16)(inp)
    x = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    return model

if "transformer" not in st.session_state:
    st.session_state.transformer = build_transformer()
    st.session_state.tx, st.session_state.ty = [], []

# ================= HELPERS =================
def auto_threshold():
    all_perf = list(st.session_state.model_perf["A"]) + list(st.session_state.model_perf["B"])
    if len(all_perf) < 5:
        return BASE_CONF
    acc = sum(all_perf) / len(all_perf)
    return int(np.clip(BASE_CONF + (0.6 - acc) * 20, 60, 75))

def regime_shift(seq):
    if len(seq) < 6:
        return False
    last = seq[-6:]
    return len(set(last)) > 1 and abs(last.count(1) - last.count(0)) <= 1

def learn_patterns(seq):
    for k in range(3,6):
        if len(seq) > k:
            key = tuple(seq[-(k+1):-1])
            st.session_state.patterns[k][key][seq[-1]] += 1

def predict_pattern(seq):
    for k in range(5,2,-1):
        key = tuple(seq[-k:])
        if key in st.session_state.patterns[k]:
            c = st.session_state.patterns[k][key]
            t = sum(c.values())
            if t >= 3:
                return ("BIG", c[1]/t) if c[1] > c[0] else ("SMALL", c[0]/t)
    return None, 0

def transformer_pred(seq):
    if len(seq) < SEQ_MAX:
        return None, 0
    x = np.array(seq[-SEQ_MAX:]).reshape(1,SEQ_MAX)
    p = float(st.session_state.transformer.predict(x, verbose=0)[0][0])
    return ("BIG" if p >= 0.5 else "SMALL"), abs(p-0.5)*2

def naive_bayes_pred(seq):
    if len(st.session_state.X_nb) < 10:
        return None, 0
    clf = MultinomialNB()
    clf.fit(st.session_state.X_nb, st.session_state.y_nb)
    p = clf.predict_proba([seq[-10:]])[0]
    idx = np.argmax(p)
    return ("BIG" if idx == 1 else "SMALL"), p[idx]

# ================= UI =================
st.title("🧠 Heavy AI Predictor (Auto-Tuned + A/B Tested)")
st.metric("Total Data", len(st.session_state.seq))

prediction, confidence = None, 0
conf_threshold = auto_threshold()
regime = regime_shift(st.session_state.seq)

if len(st.session_state.seq) >= MIN_DATA and st.session_state.loss_streak < LOSS_LIMIT:

    # --- Model A (Pattern) ---
    a_pred, a_strength = predict_pattern(st.session_state.seq)

    # --- Model B (Transformer + NB) ---
    t_pred, t_strength = transformer_pred(st.session_state.seq)
    nb_pred, nb_strength = naive_bayes_pred(st.session_state.seq)

    model_votes = []
    if a_pred:
        model_votes.append(("A", a_pred, a_strength))
    if t_pred:
        model_votes.append(("B", t_pred, t_strength))
    if nb_pred:
        model_votes.append(("B", nb_pred, nb_strength))

    if model_votes:
        votes = Counter(v[1] for v in model_votes)
        top, count = votes.most_common(1)[0]
        avg_strength = np.mean([v[2] for v in model_votes])
        confidence = int(60 + avg_strength * 40)

        if regime:
            confidence -= 10

        if count >= 1 and confidence >= conf_threshold:
            prediction = top
            st.success(f"🎯 Prediction: {prediction}")
            st.write(f"Confidence: {confidence}% (threshold {conf_threshold}%)")
            if regime:
                st.warning("⚠️ Regime shift detected — confidence reduced")
        else:
            st.warning("⏳ WAIT (low confidence)")
else:
    st.warning("🔒 PROFIT PROTECTION ACTIVE")

# ================= CONFIRM & LEARN (FIXED UI) =================
st.subheader("Confirm & Learn")
actual = st.radio("Actual Result (temporary)", ["BIG","SMALL"], horizontal=True)

if st.button("Confirm & Learn"):
    val = 1 if actual == "BIG" else 0
    st.session_state.seq.append(val)

    learn_patterns(st.session_state.seq)

    if len(st.session_state.seq) >= 10:
        st.session_state.X_nb.append(st.session_state.seq[-10:])
        st.session_state.y_nb.append(val)

    if len(st.session_state.seq) >= SEQ_MAX:
        st.session_state.tx.append(st.session_state.seq[-SEQ_MAX:])
        st.session_state.ty.append(val)
        if len(st.session_state.tx) % 5 == 0:
            X = np.array(st.session_state.tx)
            y = np.array(st.session_state.ty)
            st.session_state.transformer.fit(X,y,epochs=2,verbose=0)

    result = "WAIT"
    if prediction:
        result = "WIN" if prediction == actual else "LOSS"
        st.session_state.loss_streak = 0 if result=="WIN" else st.session_state.loss_streak+1

        for m, pred, _ in model_votes:
            st.session_state.model_perf[m].append(1 if pred==actual else 0)

    st.session_state.log.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Prediction": prediction,
        "Actual": actual,
        "Confidence": confidence,
        "Result": result
    })

    st.success(f"Saved → {result}")

# ================= HISTORY =================
st.divider()
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df, use_container_width=True)

    buf = StringIO()
    df.to_csv(buf, index=False)
    st.download_button("⬇️ Download CSV", buf.getvalue(), "heavy_ai_final.csv")
