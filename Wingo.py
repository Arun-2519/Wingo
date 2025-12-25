import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict, Counter, deque
from datetime import datetime
from io import StringIO
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ---- Naive Bayes ----
try:
    from sklearn.naive_bayes import MultinomialNB
except ModuleNotFoundError:
    st.error("❌ scikit-learn missing. Add it to requirements.txt")
    st.stop()

# ================= CONFIG =================
MIN_DATA = 10          # ⬅️ reduced
CONF_MIN = 65
LOSS_LIMIT = 2
SEQ_MAX = 5

st.set_page_config(page_title="🧠 Heavy AI Predictor (Self-Regulating)", layout="wide")

# ================= SESSION INIT =================
def init_state():
    st.session_state.seq = []
    st.session_state.patterns = {k: defaultdict(lambda: defaultdict(int)) for k in range(3,6)}
    st.session_state.log = []
    st.session_state.model_stats = {
        "pattern": deque(maxlen=10),
        "transformer": deque(maxlen=10),
        "naive_bayes": deque(maxlen=10)
    }
    st.session_state.loss_streak = 0

if "seq" not in st.session_state:
    init_state()

# ---- Naive Bayes memory ----
if "X_nb" not in st.session_state:
    st.session_state.X_nb = []
if "y_nb" not in st.session_state:
    st.session_state.y_nb = []

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
def model_enabled(name):
    stats = st.session_state.model_stats[name]
    if len(stats) < 6:
        return True
    acc = sum(stats) / len(stats)
    return acc >= 0.45

def learn_patterns(seq):
    for k in range(3,6):
        if len(seq) > k:
            key = tuple(seq[-(k+1):-1])
            nxt = seq[-1]
            st.session_state.patterns[k][key][nxt] += 1

def predict_pattern(seq):
    for k in range(5,2,-1):
        if len(seq) >= k:
            key = tuple(seq[-k:])
            if key in st.session_state.patterns[k]:
                c = st.session_state.patterns[k][key]
                t = sum(c.values())
                if t >= 3:
                    return ("BIG", c[1]/t) if c[1] > c[0] else ("SMALL", c[0]/t)
    return None, 0

def transformer_prob(seq):
    if len(seq) < SEQ_MAX:
        return 0.5
    x = np.array(seq[-SEQ_MAX:]).reshape(1,SEQ_MAX)
    return float(st.session_state.transformer.predict(x, verbose=0)[0][0])

def naive_bayes_prob(seq):
    if len(st.session_state.X_nb) < 10 or len(seq) < 10:
        return None, 0
    clf = MultinomialNB()
    clf.fit(st.session_state.X_nb, st.session_state.y_nb)
    p = clf.predict_proba([seq[-10:]])[0]
    idx = np.argmax(p)
    return ("BIG" if idx == 1 else "SMALL"), p[idx]

# ================= UI =================
st.title("🧠 Heavy AI Predictor (Auto-Regulating System)")
st.metric("Total Data Learned", len(st.session_state.seq))

prediction, confidence = None, 0
signals = []

if len(st.session_state.seq) >= MIN_DATA and st.session_state.loss_streak < LOSS_LIMIT:

    # --- Pattern AI ---
    if model_enabled("pattern"):
        p_pred, p_strength = predict_pattern(st.session_state.seq)
        if p_pred:
            signals.append(("pattern", p_pred, p_strength))

    # --- Transformer ---
    if model_enabled("transformer"):
        t_prob = transformer_prob(st.session_state.seq)
        if t_prob >= 0.56 or t_prob <= 0.44:
            signals.append(("transformer", "BIG" if t_prob >= 0.5 else "SMALL", abs(t_prob-0.5)))

    # --- Naive Bayes ---
    if model_enabled("naive_bayes"):
        nb_pred, nb_prob = naive_bayes_prob(st.session_state.seq)
        if nb_pred:
            signals.append(("naive_bayes", nb_pred, nb_prob))

    if len(signals) >= 2:
        votes = [s[1] for s in signals]
        prediction = Counter(votes).most_common(1)[0][0]

        confidence = int(60 + sum(s[2] for s in signals)/len(signals) * 40)

        if confidence >= CONF_MIN:
            st.success(f"🎯 Prediction: {prediction}")
            st.write(f"Confidence: {confidence}%")
            st.write("Active Models:", [s[0] for s in signals])
        else:
            prediction = None
            st.warning("⏳ WAIT (Low confidence)")
    else:
        st.warning("⏳ WAIT (Not enough strong models)")
else:
    st.warning("🔒 PROFIT PROTECTION ACTIVE (HARD WAIT)")

# ================= CONFIRM & LEARN =================
st.subheader("Confirm & Learn")
actual = st.radio("Actual Result", ["BIG","SMALL"], horizontal=True)

if st.button("Confirm & Learn"):
    val = 1 if actual == "BIG" else 0
    st.session_state.seq.append(val)

    # Learn ordered patterns
    learn_patterns(st.session_state.seq)

    # Learn Naive Bayes
    if len(st.session_state.seq) >= 10:
        st.session_state.X_nb.append(st.session_state.seq[-10:])
        st.session_state.y_nb.append(val)

    # Train Transformer
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
        st.session_state.loss_streak = 0 if result=="WIN" else st.session_state.loss_streak + 1

        # Update model stats
        for name, pred, _ in signals:
            st.session_state.model_stats[name].append(1 if pred==actual else 0)

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
st.subheader("📄 Prediction History")

if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df, use_container_width=True)

    buf = StringIO()
    df.to_csv(buf, index=False)
    st.download_button("⬇️ Download CSV", buf.getvalue(), "auto_regulated_ai.csv")
