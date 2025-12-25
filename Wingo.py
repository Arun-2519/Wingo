import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
from io import StringIO
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ================= CONFIG =================
MIN_DATA = 20
SEQ_MIN = 3
SEQ_MAX = 5
WAIT_LOSS_LIMIT = 2

st.set_page_config(page_title="AI Wingo Predictor (Heavy AI)", layout="wide")

# ================= SESSION INIT =================
def reset_session(keep_knowledge=True):
    if not keep_knowledge:
        st.session_state.seq = []
        st.session_state.patterns = {k: defaultdict(lambda: defaultdict(int)) for k in range(3,6)}
    st.session_state.log = []
    st.session_state.wait_level = 0

if "seq" not in st.session_state:
    st.session_state.seq = []
if "patterns" not in st.session_state:
    st.session_state.patterns = {k: defaultdict(lambda: defaultdict(int)) for k in range(3,6)}
if "log" not in st.session_state:
    st.session_state.log = []
if "wait_level" not in st.session_state:
    st.session_state.wait_level = 0

# ================= TRANSFORMER MODEL =================
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

# ================= LEARN ORDERED PATTERNS =================
def learn_patterns(seq):
    for k in range(3,6):
        if len(seq) > k:
            key = tuple(seq[-(k+1):-1])
            nxt = seq[-1]
            st.session_state.patterns[k][key][nxt] += 1

# ================= PREDICT ORDERED PATTERNS =================
def predict_pattern(seq):
    for k in range(5,2,-1):
        if len(seq) >= k:
            key = tuple(seq[-k:])
            if key in st.session_state.patterns[k]:
                counts = st.session_state.patterns[k][key]
                total = sum(counts.values())
                if total >= 3:
                    if counts[1] > counts[0]:
                        return "BIG", counts[1]/total
                    if counts[0] > counts[1]:
                        return "SMALL", counts[0]/total
    return None, 0

# ================= REVERSAL DETECTOR =================
def reversal_detect(seq):
    if len(seq) < 4:
        return False
    last = seq[-4:]
    if last.count(last[0]) == 3 and last[-1] != last[0]:
        return True
    return False

# ================= TRANSFORMER PROB =================
def transformer_prob(seq):
    if len(seq) < SEQ_MAX:
        return 0.5
    x = np.array(seq[-SEQ_MAX:]).reshape(1,SEQ_MAX)
    return float(st.session_state.transformer.predict(x, verbose=0)[0][0])

# ================= UI =================
st.title("🧠 AI Wingo Predictor — Heavy Intelligence System")
st.metric("Total Data Learned", len(st.session_state.seq))

prediction, confidence = None, 0
reversal = reversal_detect(st.session_state.seq)

if len(st.session_state.seq) >= MIN_DATA:
    p_pred, p_strength = predict_pattern(st.session_state.seq)
    t_prob = transformer_prob(st.session_state.seq)

    signals = 0
    if p_pred: signals += 1
    if t_prob > 0.55 or t_prob < 0.45: signals += 1
    if not reversal: signals += 1

    final_score = (p_strength + abs(t_prob-0.5)) / 2
    confidence = int(60 + final_score * 40)

    if signals >= 3 and confidence >= 60 and not reversal and st.session_state.wait_level < WAIT_LOSS_LIMIT:
        prediction = p_pred
        st.success(f"🎯 Prediction: {prediction}")
        st.write(f"Confidence: {confidence}%")
        st.write(f"Transformer prob: {t_prob:.2f}")
    else:
        st.warning("⏳ SMART WAIT (risk detected)")
else:
    st.info("⏳ Learning phase…")

# ================= CONFIRM & LEARN =================
st.subheader("Confirm & Learn")
actual = st.radio("Actual Result", ["BIG","SMALL"], horizontal=True)

if st.button("Confirm & Learn"):
    val = 1 if actual=="BIG" else 0
    st.session_state.seq.append(val)

    learn_patterns(st.session_state.seq)

    if len(st.session_state.seq) >= SEQ_MAX:
        st.session_state.tx.append(st.session_state.seq[-SEQ_MAX:])
        st.session_state.ty.append(val)
        if len(st.session_state.tx) % 5 == 0:
            X = np.array(st.session_state.tx)
            y = np.array(st.session_state.ty)
            st.session_state.transformer.fit(X,y,epochs=2,verbose=0)

    result = "WAIT"
    if prediction:
        result = "WIN" if prediction==actual else "LOSS"
        st.session_state.wait_level = 0 if result=="WIN" else st.session_state.wait_level+1

    st.session_state.log.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Prediction": prediction,
        "Actual": actual,
        "Confidence": confidence,
        "Result": result
    })

    st.success(f"Saved → {result}")

# ================= PATTERN VISUALIZATION =================
st.divider()
st.subheader("📊 Pattern Frequency Visualization")

freq = Counter()
for k in st.session_state.patterns:
    for pat in st.session_state.patterns[k]:
        freq[str(pat)] += sum(st.session_state.patterns[k][pat].values())

if freq:
    dfp = pd.DataFrame(freq.most_common(10), columns=["Pattern","Frequency"])
    st.dataframe(dfp)

# ================= CSV PREVIEW =================
st.divider()
st.subheader("📄 Prediction History")

if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df, use_container_width=True)

    col1,col2,col3 = st.columns([6,1,1])
    with col3:
        buf = StringIO()
        df.to_csv(buf,index=False)
        st.download_button("⬇️ Download CSV", buf.getvalue(), "wingo_heavy_ai.csv")
