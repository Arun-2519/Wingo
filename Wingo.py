import streamlit as st
import pandas as pd
import math, json, os
from collections import defaultdict
from io import BytesIO
from datetime import date

# ================= CONFIG =================
MIN_TOTAL_DATA = 10
DAILY_WARMUP = 5
MIN_EDGE = 0.25          # net edge threshold
MIN_WINRATE = 0.60
MIN_PATTERN_COUNT = 6
POST_LOSS_WAIT = 1

DATA_DIR = "ai_memory"
HISTORY_FILE = f"{DATA_DIR}/history.json"
PATTERN_FILE = f"{DATA_DIR}/pattern_stats.json"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config("🔵🔴 BIG vs SMALL AI", layout="centered")
st.title("🔵 BIG vs 🔴 BIG–SMALL AI Predictor (DE-BIASED & STABLE)")

# ================= LOAD =================
def load_json(p, d):
    return json.load(open(p)) if os.path.exists(p) else d

def save_json(p, d):
    json.dump(d, open(p,"w"))

if "history" not in st.session_state:
    st.session_state.history = load_json(HISTORY_FILE, [])

if "pattern_stats" not in st.session_state:
    raw = load_json(PATTERN_FILE, {})
    st.session_state.pattern_stats = defaultdict(lambda: defaultdict(lambda:[0,0]))
    for k,v in raw.items():
        st.session_state.pattern_stats[tuple(eval(k))] = v

if "today_inputs" not in st.session_state:
    st.session_state.today_inputs = 0
if "today_date" not in st.session_state:
    st.session_state.today_date = str(date.today())
if "post_loss_wait" not in st.session_state:
    st.session_state.post_loss_wait = 0

if st.session_state.today_date != str(date.today()):
    st.session_state.today_date = str(date.today())
    st.session_state.today_inputs = 0

# ================= HELPERS =================
def entropy(seq):
    p = seq.count("BIG") / len(seq)
    if p in (0,1): return 0
    return -p*math.log2(p)-(1-p)*math.log2(1-p)

def detect_chop(seq):
    return len(seq) < 6 or entropy(seq[-6:]) > 0.9

def patterns(seq):
    return [tuple(seq[-k:]) for k in range(4,9) if len(seq)>=k]

def net_edge(win, tot):
    return (win - (tot - win)) / tot

# ================= PREDICTION =================
prediction, confidence = None, 0
hist = st.session_state.history

if st.session_state.post_loss_wait > 0:
    st.warning("⏳ WAIT (post-loss stabilization)")

elif len(hist) >= MIN_TOTAL_DATA and st.session_state.today_inputs >= DAILY_WARMUP:

    if detect_chop(hist):
        st.warning("⏳ WAIT (choppy regime)")
    else:
        candidates = []

        for pat in patterns(hist):
            stats = st.session_state.pattern_stats.get(pat,{})
            for res,(win,tot) in stats.items():
                if tot >= MIN_PATTERN_COUNT:
                    wr = win/tot
                    edge = net_edge(win,tot)
                    if wr>=MIN_WINRATE and edge>=MIN_EDGE:
                        candidates.append((res,wr,edge,len(pat)))

        if candidates:
            # prefer high edge, then longer pattern
            best = max(candidates, key=lambda x:(x[2],x[3]))
            prediction = best[0]
            confidence = int(best[1]*100)
            st.success(f"🎯 Prediction: {prediction}")
            st.write(f"Confidence: {confidence}% | Edge: {round(best[2],2)} | Len: {best[3]}")
        else:
            st.warning("⏳ WAIT (no high-edge pattern)")
else:
    st.warning("🕐 Warming up…")

# ================= INPUT =================
actual = st.selectbox("Enter Actual Result", ["BIG","SMALL"])

if st.button("Confirm & Learn"):

    if st.session_state.post_loss_wait>0:
        st.session_state.post_loss_wait-=1

    st.session_state.history.append(actual)
    st.session_state.today_inputs+=1

    for pat in patterns(st.session_state.history[:-1]):
        if prediction:
            st.session_state.pattern_stats[pat][prediction][1]+=1
            if prediction==actual:
                st.session_state.pattern_stats[pat][prediction][0]+=1

    if prediction and prediction!=actual:
        st.session_state.post_loss_wait=POST_LOSS_WAIT

    save_json(HISTORY_FILE, st.session_state.history)
    save_json(PATTERN_FILE,{str(k):v for k,v in st.session_state.pattern_stats.items()})

    st.success("Saved & Learned")
    st.rerun()

# ================= VIEW =================
if st.session_state.history:
    df = pd.DataFrame({"Result":st.session_state.history})
    st.dataframe(df.tail(50))

st.caption("Edge-based AI • Long pattern priority • No blind alternation")
