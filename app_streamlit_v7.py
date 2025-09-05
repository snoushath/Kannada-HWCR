# app_streamlit_v7.py
import streamlit as st
from pathlib import Path
import json, shutil, time
import numpy as np
import cv2
from akshara_lib_v7 import load_cache_and_indices, query_once, preprocess

st.set_page_config(page_title="Akshara Hybrid Search", layout="wide")
st.sidebar.header("Load Cache")
cache_root = st.sidebar.text_input("Cache folder (must contain features.pkl)", "")
if "data" not in st.session_state: st.session_state.data=None
if st.sidebar.button("Load cache"):
    p = Path(cache_root)
    if not (p.exists() and (p/"features.pkl").exists()):
        st.error(f"Cache not found: {p/'features.pkl'}")
    else:
        st.session_state.data = load_cache_and_indices(p, backend="auto")
        st.success("Cache loaded.")

st.title("Akshara Hybrid Search (Upload → Matches + JSON)")
st.caption("HOG/LBP/Hu + Tversky (base-aware) + Topology + Composition/OT. ANN per base.")

if st.session_state.data is None:
    st.info("Load a cache first from the sidebar to start.")
    st.stop()

# controls
st.sidebar.header("Retrieval Settings")
neighbors = st.sidebar.slider("Neighbors K (coarse retrieve)", 50, 500, 200, 10)
cap = st.sidebar.slider("Per-class cap (top-M in vote)", 1, 10, 3, 1)
st.sidebar.header("Fusion Weights")
w_hog  = st.sidebar.number_input("w_hog", 0.0, 1.0, 0.10, 0.01)
w_tv   = st.sidebar.number_input("w_tv",  0.0, 1.0, 0.25, 0.01)
w_topo = st.sidebar.number_input("w_topo",0.0, 1.0, 0.20, 0.01)
w_comp = st.sidebar.number_input("w_comp",0.0, 1.0, 0.35, 0.01)
w_cent = st.sidebar.number_input("w_cent",0.0, 1.0, 0.05, 0.01)
temp   = st.sidebar.number_input("softmax temp", 0.01, 1.0, 0.25, 0.01)
alpha  = st.sidebar.number_input("alpha (Tversky)", 0.0, 1.0, 0.7, 0.05)
beta   = st.sidebar.number_input("beta  (Tversky)", 0.0, 1.0, 0.3, 0.05)

st.sidebar.header("Session log")
log_dir = Path(st.sidebar.text_input("Run dir (JSONL + thumbs)", "runs/session_"+time.strftime("%Y%m%d")))
log_dir.mkdir(parents=True, exist_ok=True)
(st.session_state.setdefault("thumb_dir", log_dir/"thumbs")).mkdir(parents=True, exist_ok=True)
log_path = log_dir/"session_log.jsonl"

# uploader
colL, colR = st.columns([1,2])
with colL:
    up = st.file_uploader("Upload a query image", type=["png","jpg","jpeg","bmp","tif","tiff","webp"])
    if up:
        bytes_ = up.read()
        arr = np.frombuffer(bytes_, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            st.error("Could not decode image."); st.stop()
        # write temp file to keep path for logs
        qtmp = log_dir/f"query_{int(time.time()*1000)}.png"
        cv2.imwrite(str(qtmp), img)
        query_path = qtmp
    else:
        st.stop()

with colR:
    st.subheader("Results")
    out = query_once(
        st.session_state.data, Path(query_path),
        neighbors_k=neighbors, cap=cap,
        w_hog=w_hog, w_tv=w_tv, w_topo=w_topo, w_comp=w_comp, w_cent=w_cent,
        temp=temp, alpha=alpha, beta=beta
    )
    st.code(json.dumps(out, indent=2, ensure_ascii=False), language="json")

    # thumbs
    def show_thumb(p):
        im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        st.image(im, clamp=True, caption=str(Path(p).name))
    st.markdown("**Top matches**")
    cols = st.columns(5)
    for i, nn in enumerate(out["nearest_neighbors"]):
        with cols[i%5]:
            show_thumb(nn["file"])

    if st.button("Save to session_log.jsonl"):
        # copy thumbs
        qdst = st.session_state.thumb_dir/f"q_{Path(query_path).name}"
        shutil.copy2(query_path, qdst)
        for nn in out["nearest_neighbors"]:
            src = nn["file"]; dst = st.session_state.thumb_dir/Path(src).name
            if not Path(dst).exists():
                try: shutil.copy2(src, dst)
                except Exception: pass
        # append JSONL
        rec = dict(out); rec["thumb_dir"]=str(st.session_state.thumb_dir); rec["saved_at"]=time.ctime()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
        st.success(f"Saved to {log_path}")
