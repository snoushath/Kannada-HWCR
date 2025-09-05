
# import json, re, time, warnings, tempfile, os, sys
# from pathlib import Path
# from typing import Dict, Any, Tuple, List
# import numpy as np
# import cv2
# from joblib import dump, load
# from collections import defaultdict
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
# from skimage.feature import hog, local_binary_pattern
# from skimage.morphology import skeletonize
# from skimage.measure import euler_number
# from pathlib import Path
# from joblib import load

# import argparse, json, sys
# from pathlib import Path
# from akshara_lib_v7 import load_cache_and_indices, query_once

# # import argparse, json, sys
# # from pathlib import Path
# # from akshara_lib_v7 import build_cache, load_cache_and_indices, query_once
# warnings.filterwarnings("ignore", category=UserWarning)
# SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# HAS_FAISS = False
# HAS_HNSW  = False
# try:
#     import faiss
#     HAS_FAISS = True
# except Exception:
#     pass
# try:
#     import hnswlib
#     HAS_HNSW = True
# except Exception:
#     pass

# # --------------------- I/O utils ---------------------
# def imread_unicode(path, flags=cv2.IMREAD_COLOR):
#     p = Path(path)
#     try:
#         data = p.read_bytes()
#         arr = np.frombuffer(data, dtype=np.uint8)
#         if arr.size:
#             img = cv2.imdecode(arr, flags)
#             if img is not None:
#                 return img
#     except Exception:
#         pass
#     try:
#         suffix = p.suffix or ".png"
#         with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
#             tmp.write(p.read_bytes()); tmp_path = tmp.name
#         img = cv2.imread(tmp_path, flags)
#         try: Path(tmp_path).unlink(missing_ok=True)
#         except Exception: pass
#         return img
#     except Exception:
#         return None

# def is_image_file(p: Path) -> bool:
#     try: return p.suffix.lower() in SUPPORTED_EXTS
#     except: return False

# def list_class_image_paths(dataset_dir: Path):
#     pairs = []
#     subs = [p for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
#     if subs:
#         for base_dir in sorted(subs):
#             for root, _, files in os.walk(base_dir):
#                 rp = Path(root)
#                 if any(part.startswith('.') for part in rp.parts): continue
#                 label = rp.name if rp != base_dir else base_dir.name
#                 for f in files:
#                     p = rp / f
#                     if is_image_file(p): pairs.append((label, str(p)))
#         if pairs: return pairs
#     # flat
#     tok = re.compile(r'[_\-\s\.]+'); dig = re.compile(r'(\d+)'); word = re.compile(r'^[^\W\d_]+$', re.UNICODE)
#     flats = [p for p in dataset_dir.iterdir() if p.is_file() and is_image_file(p)]
#     if flats:
#         def infer(stem):
#             t = tok.split(stem)[0]
#             if not t: return None
#             t = dig.split(t)[0]
#             return t if word.match(t) else None
#         for p in flats:
#             cls = infer(p.stem) or "all"
#             pairs.append((cls, str(p)))
#         return pairs
#     raise RuntimeError(f"No images under: {dataset_dir}")

# def base_from_path(dataset_dir: Path, file_path: str|Path) -> str:
#     p = Path(file_path)
#     try:
#         rel = p.relative_to(dataset_dir)
#         return rel.parts[0] if len(rel.parts) >= 1 else "all"
#     except Exception:
#         return p.parent.name or "all"

# # --------------------- preprocessing ---------------------
# def crop_to_ink(binary_img: np.ndarray, margin=3):
#     mask = (binary_img < 128).astype(np.uint8)
#     ys, xs = np.where(mask > 0)
#     if ys.size == 0: return binary_img
#     y0 = max(int(ys.min())-margin, 0); y1 = min(int(ys.max())+1+margin, binary_img.shape[0])
#     x0 = max(int(xs.min())-margin, 0); x1 = min(int(xs.max())+1+margin, binary_img.shape[1])
#     return binary_img[y0:y1, x0:x1]

# def preprocess(path: str, size=192) -> np.ndarray:
#     img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
#     if img is None: raise RuntimeError(f"read fail: {path}")
#     if float(img.mean()) > 127: img = 255 - img
#     img = cv2.equalizeHist(img)
#     _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     k = np.ones((3,3), np.uint8)
#     th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, 1)
#     th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, 1)
#     if th.mean() < 127: th = 255 - th
#     th = crop_to_ink(th, 3)
#     h,w = th.shape[:2]; dim=max(h,w)
#     pt=(dim-h)//2; pb=dim-h-pt; pl=(dim-w)//2; pr=dim-w-pl
#     th = cv2.copyMakeBorder(th, pt,pb,pl,pr, cv2.BORDER_CONSTANT, 255)
#     th = cv2.resize(th, (size,size), cv2.INTER_AREA)
#     return th

# # --------------------- classic features ---------------------
# def hu_feature(img):
#     m=cv2.moments(img); hu=cv2.HuMoments(m).flatten()
#     return (-np.sign(hu)*np.log10(np.abs(hu)+1e-12)).astype(np.float32)

# def hog_feature(img):
#     f = hog(img, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2),
#             block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)
#     return f.astype(np.float32)

# def lbp_feature(img, P=8, R=2):
#     lbp = local_binary_pattern(img, P=P, R=R, method="uniform")
#     n_bins=P+2; edges=np.arange(0,n_bins+1,1,dtype=np.int32)
#     hist,_ = np.histogram(lbp.ravel(), bins=edges, density=True)
#     return hist.astype(np.float32)

# def classical_vec(proc_img): 
#     return np.concatenate([hu_feature(proc_img), hog_feature(proc_img), lbp_feature(proc_img)], 0)

# # --------------------- bit-pack + Tversky ---------------------
# def pack_mask(proc_img: np.ndarray) -> Tuple[np.ndarray, int]:
#     mask = (proc_img < 128).astype(np.uint8).reshape(-1)
#     return np.packbits(mask), int(mask.size)

# def unpack_mask(packed: np.ndarray, bitlen: int, size: int) -> np.ndarray:
#     u = np.unpackbits(packed)[:bitlen].astype(bool)
#     return u.reshape(size, size)

# _POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
# def tversky_bits(A: np.ndarray, B: np.ndarray, bitlen: int, alpha=0.7, beta=0.3) -> float:
#     A = A.astype(np.uint8, copy=False); B = B.astype(np.uint8, copy=False)
#     inter = int(_POP[np.bitwise_and(A,B)].sum())
#     onlyA = int(_POP[np.bitwise_and(A, np.bitwise_not(B))].sum())
#     onlyB = int(_POP[np.bitwise_and(B, np.bitwise_not(A))].sum())
#     denom = inter + alpha*onlyA + beta*onlyB + 1e-12
#     return float(inter/denom)

# # --------------------- topology features ---------------------
# def _neighbors8(y,x,H,W):
#     for dy in (-1,0,1):
#         for dx in (-1,0,1):
#             if dy==0 and dx==0: continue
#             yy,xx=y+dy,x+dx
#             if 0<=yy<H and 0<=xx<W: yield yy,xx

# def topo_features(proc_img: np.ndarray) -> np.ndarray:
#     fg = (proc_img<128).astype(np.uint8)
#     if fg.sum()==0:
#         return np.zeros(16, dtype=np.float32)
#     sk = skeletonize(fg>0).astype(np.uint8)
#     H,W = sk.shape
#     deg = np.zeros_like(sk, dtype=np.uint8)
#     ys,xs = np.where(sk>0)
#     for y,x in zip(ys,xs):
#         c=0
#         for yy,xx in _neighbors8(y,x,H,W):
#             if sk[yy,xx]: c+=1
#         deg[y,x]=c
#     endpoints = int((deg==1).sum())
#     junctions = int((deg>=3).sum())
#     pathpx    = int((deg==2).sum())
#     total     = int(sk.sum())
#     loops     = int(max(0, -euler_number(sk, connectivity=2)))
#     # rough edge-length histogram
#     node_mask = (deg!=2) & (sk>0)
#     visited=set(); lengths=[]
#     for y,x in zip(*np.where(node_mask)):
#         for yy,xx in _neighbors8(y,x,H,W):
#             if not sk[yy,xx]: continue
#             prev=(y,x); cur=(yy,xx); L=1
#             while sk[cur] and deg[cur]==2:
#                 nxt=None
#                 for z in _neighbors8(cur[0],cur[1],H,W):
#                     if sk[z] and z!=prev: nxt=z; break
#                 if nxt is None: break
#                 prev=cur; cur=nxt; L+=1
#             lengths.append(L)
#     bins = [0,4,8,16,10**9]
#     hist = np.histogram(lengths, bins=bins)[0].astype(np.float32)
#     d_hist = np.array([ (deg==k).sum() for k in (0,1,2,3) ], dtype=np.float32)
#     vec = np.concatenate([np.array([endpoints, junctions, pathpx, total, loops], dtype=np.float32),
#                           d_hist, hist], 0)
#     if vec.sum()>0: vec = vec/(np.linalg.norm(vec)+1e-12)
#     return vec.astype(np.float32)

# # --------------------- composition (zones + OT) ---------------------
# def zone_slices(H,W, r=3, c=3):
#     ys = [ (i*H)//r for i in range(r) ] + [H]
#     xs = [ (j*W)//c for j in range(c) ] + [W]
#     return [(slice(ys[i],ys[i+1]), slice(xs[j],xs[j+1])) for i in range(r) for j in range(c)]

# _ZS_CACHE={}
# def _zs(H,W):
#     key=(H,W)
#     if key not in _ZS_CACHE: _ZS_CACHE[key]=zone_slices(H,W,3,3)
#     return _ZS_CACHE[key]

# def zone_stats_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     H,W = mask.shape
#     zs = _zs(H,W)
#     masses=[]; cents=[]
#     yy,xx = np.mgrid[0:H,0:W]
#     for ys,xs in zs:
#         z = mask[ys,xs]
#         m = float(z.sum())
#         masses.append(m)
#         if m>0:
#             y = float((yy[ys,xs]*z).sum())/m
#             x = float((xx[ys,xs]*z).sum())/m
#         else:
#             y = (ys.start+ys.stop)/2.0; x=(xs.start+xs.stop)/2.0
#         cents.append([y/H, x/W])
#     masses = np.array(masses, dtype=np.float32)
#     s = masses.sum()
#     if s>0: masses = masses/s
#     cents  = np.array(cents, dtype=np.float32)
#     return masses, cents

# def _cdist_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     # a: (n,2), b: (m,2)
#     diff = a[:,None,:] - b[None,:,:]
#     return np.sqrt(np.sum(diff*diff, axis=-1))

# def zones_tversky(qmask: np.ndarray, cmask: np.ndarray, alpha=0.7, beta=0.3) -> float:
#     zs = _zs(*qmask.shape)
#     tvs=[]
#     for ys,xs in zs:
#         qz = qmask[ys,xs]; cz = cmask[ys,xs]
#         inter = float((qz & cz).sum())
#         onlyA = float((qz & (~cz)).sum())
#         onlyB = float(((~qz) & cz).sum())
#         tvs.append(inter/(inter+alpha*onlyA+beta*onlyB+1e-12))
#     return float(np.mean(tvs))

# def sinkhorn_ot(m1,x1, m2,x2, reg=0.1, iters=50):
#     # m1,m2 sum to 1; x1:(n,2), x2:(m,2) in [0,1]
#     C = _cdist_euclidean(x1, x2)                 # [0, sqrt(2)]
#     K = np.exp(-(C/reg))
#     u = np.ones_like(m1)/len(m1)
#     v = np.ones_like(m2)/len(m2)
#     for _ in range(iters):
#         u = m1/(K.dot(v)+1e-12)
#         v = m2/(K.T.dot(u)+1e-12)
#     T = (u[:,None]*K)*v[None,:]
#     cost = float((T*C).sum() / np.sqrt(2.0))     # normalize to [0,1]
#     return cost

# # --------------------- helpers ---------------------
# def softmax(x, temperature=1.0):
#     x=np.array(x,dtype=float); x/=float(temperature); x-=x.max()
#     e=np.exp(x); return e/(e.sum()+1e-12)

# def minmax01(a):
#     a=np.asarray(a,dtype=float); mn, mx=a.min(), a.max()
#     return np.zeros_like(a) if mx-mn<1e-12 else (a-mn)/(mx-mn)

# # --------------------- cache build/load (pure sklearn) ---------------------
# def build_cache(dataset_dir: Path, cache_dir: Path, size=192, force_rebuild=False):
#     cache_dir.mkdir(parents=True, exist_ok=True)
#     pkl = cache_dir/"features.pkl"
#     if pkl.exists() and not force_rebuild:
#         print(f"[INFO] Using existing cache: {pkl}"); return

#     pairs = list_class_image_paths(dataset_dir)
#     labels=[]; files=[]; bases=[]
#     X_hog=[]; X_topo=[]; ink_bits=[]
#     t0=time.time(); print(f"[BUILD] Indexing {len(pairs)} from {dataset_dir}")
#     for i,(cls,f) in enumerate(pairs,1):
#         try:
#             proc = preprocess(f, size=size)
#             X_hog.append(classical_vec(proc))
#             X_topo.append(topo_features(proc))
#             pb, _ = pack_mask(proc); ink_bits.append(pb)
#             labels.append(cls); files.append(f); bases.append(base_from_path(dataset_dir,f))
#         except Exception as e:
#             print(f"[WARN] skip {f}: {e}")
#         if i%400==0: print(f"  {i}/{len(pairs)}")

#     if not X_hog: raise RuntimeError("no samples")

#     X_hog = np.vstack(X_hog).astype(np.float32)
#     X_topo= np.vstack(X_topo).astype(np.float32)
#     labels=np.array(labels); files=np.array(files); bases=np.array(bases)
#     ink_bits=np.array(ink_bits, dtype=object)

#     scaler_hog = StandardScaler().fit(X_hog)
#     scaler_top = StandardScaler(with_mean=False, with_std=True).fit(X_topo)
#     X_hog_std = scaler_hog.transform(X_hog).astype(np.float32)
#     X_top_std = scaler_top.transform(X_topo).astype(np.float32)

#     # HOG class centroids
#     centroids={}
#     for cls in sorted(set(labels)):
#         idx = np.where(labels==cls)[0]
#         mu = X_hog_std[idx].mean(0); mu = mu/(np.linalg.norm(mu)+1e-12)
#         centroids[cls]=mu.astype(np.float32)

#     data = dict(
#         X_hog_std=X_hog_std, X_top_std=X_top_std,
#         scaler_hog=scaler_hog, scaler_top=scaler_top,
#         labels=labels, files=files, bases=bases,
#         ink_bits=ink_bits, size=int(size), bitlen=int(size*size),
#         centroids_hog=centroids, dataset_dir=str(dataset_dir),
#         built_at=time.ctime()
#     )
#     dump(data, pkl)
#     print(f"[OK] Saved cache -> {pkl} (N={len(labels)})  in {time.time()-t0:.1f}s")

# # def _build_per_base_indices(data: Dict[str,Any]):
# #     """Pure sklearn per-base brute-force cosine indices."""
# #     vecs = data["X_hog_std"]
# #     bases = data["bases"]
# #     base_to = {}
# #     for b in sorted(set(bases)):
# #         idx = np.where(bases==b)[0]
# #         V = vecs[idx]
# #         nn = NearestNeighbors(metric='cosine', algorithm='brute')
# #         nn.fit(V)
# #         base_to[b] = {"idx": idx, "index": nn, "vecs": V, "backend": "sk"}
# #     return base_to


# # --- REPLACE your existing _build_per_base_indices with this version ---
# def _build_per_base_indices(data, backend: str = "auto"):
#     """
#     Build one ANN index per base for the HOG-standardized features.
#     backend: "auto" | "faiss" | "hnsw" | "sk"
#     """
#     import numpy as np
#     from sklearn.neighbors import NearestNeighbors

#     V_all   = data["X_hog_std"]   # (N, D)
#     bases   = data["bases"]
#     base_to = {}

#     # resolve backend choice
#     if backend == "auto":
#         if HAS_FAISS:
#             chosen = "faiss"
#         elif HAS_HNSW:
#             chosen = "hnsw"
#         else:
#             chosen = "sk"
#     else:
#         chosen = backend
#         if chosen == "faiss" and not HAS_FAISS:
#             print("[WARN] FAISS not available → falling back to scikit-learn.")
#             chosen = "sk"
#         if chosen == "hnsw" and not HAS_HNSW:
#             print("[WARN] hnswlib not available → falling back to scikit-learn.")
#             chosen = "sk"

#     for b in sorted(set(bases)):
#         idx = np.where(bases == b)[0]
#         V   = V_all[idx]
#         D   = V.shape[1]

#         entry = {"idx": idx, "backend": chosen, "index": None, "vecs": V}

#         if chosen == "faiss":
#             # cosine via inner product on L2-normalized vectors
#             xb = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
#             index = faiss.IndexFlatIP(D)
#             index.add(xb.astype(np.float32))
#             entry["index"] = index

#         elif chosen == "hnsw":
#             p = hnswlib.Index(space="cosine", dim=D)
#             p.init_index(max_elements=len(V), ef_construction=200, M=16)
#             p.add_items(V.astype(np.float32))
#             p.set_ef(200)
#             entry["index"] = p

#         else:  # "sk"
#             nn = NearestNeighbors(metric="cosine", algorithm="brute")
#             nn.fit(V)
#             entry["index"] = nn

#         base_to[b] = entry

#     return base_to


# # def load_cache_and_indices(cache_dir: Path):
# #     data = load(cache_dir/"features.pkl")
# #     data["indices"] = _build_per_base_indices(data)
# #     return data

# # def load_cache_and_indices(cache_dir: Path, backend: str = "auto"):
# #     """Load features.pkl and build per-base ANN indices with the chosen backend."""
# #     data = load(Path(cache_dir) / "features.pkl")
# #     # _build_per_base_indices should already exist in this file
# #     data["indices"] = _build_per_base_indices(data, backend=backend)
# #     return data


# def load_cache_and_indices(cache_dir: Path, backend: str = "auto"):
#     """
#     Load features.pkl and build per-base ANN indices with the chosen backend.
#     """
#     data = load(Path(cache_dir) / "features.pkl")
#     data["indices"] = _build_per_base_indices(data, backend=backend)
#     return data

# # --------------------- query ---------------------
# def query_once(data: Dict[str,Any], query_path: Path,
#                neighbors_k=200, cap=3,
#                w_hog=0.10, w_tv=0.25, w_topo=0.20, w_comp=0.35, w_cent=0.05,
#                temp=0.25, alpha=0.7, beta=0.3):
#     size = int(data["size"]); bitlen=int(data["bitlen"])
#     # query preprocess and features
#     proc = preprocess(str(query_path), size=size)
#     q_hog = classical_vec(proc)
#     q_hog_std = data["scaler_hog"].transform(q_hog.reshape(1,-1)).astype(np.float32)[0]
#     q_top = topo_features(proc)
#     q_top_std = data["scaler_top"].transform(q_top.reshape(1,-1)).astype(np.float32)[0]
#     q_bits, _ = pack_mask(proc)
#     q_mask = unpack_mask(q_bits, bitlen, size)

#     # per-base index
#     q_base = base_from_path(Path(data["dataset_dir"]), query_path)
#     entry = data["indices"].get(q_base, None)
#     if entry is None:    # fallback to any base present
#         entry = list(data["indices"].values())[0]
#     idx_pool = entry["idx"]
#     nn = entry["index"]

#     # coarse kNN (cosine → distances; we turn to similarity)
#     K = min(neighbors_k, len(idx_pool))
#     dists, loc = nn.kneighbors(q_hog_std.reshape(1,-1), n_neighbors=K)
#     loc = loc[0]; sims_primary = 1.0 - dists[0]
#     cand = idx_pool[loc]

#     labels = data["labels"]; files = data["files"]; bases = data["bases"]
#     X_hog_std = data["X_hog_std"]; X_top_std = data["X_top_std"]
#     ink_bits_db = data["ink_bits"]; cents = data["centroids_hog"]

#     # compute channels for candidates
#     sims_hog = np.array([ float(np.dot(q_hog_std, X_hog_std[i]) /
#                                 ((np.linalg.norm(q_hog_std)+1e-12)*(np.linalg.norm(X_hog_std[i])+1e-12)))
#                          for i in cand ], dtype=np.float32)
#     sims_top = np.array([ float(np.dot(q_top_std, X_top_std[i]) /
#                                 ((np.linalg.norm(q_top_std)+1e-12)*(np.linalg.norm(X_top_std[i])+1e-12)))
#                          for i in cand ], dtype=np.float32)

#     # full Tversky and composition
#     sims_tv  = np.empty(len(cand), dtype=np.float32)
#     sims_comp= np.empty(len(cand), dtype=np.float32)
#     q_mass, q_cent = zone_stats_from_mask(q_mask)
#     for k, gi in enumerate(cand):
#         tv = tversky_bits(q_bits, ink_bits_db[gi], bitlen, alpha=alpha, beta=beta)
#         cmask = unpack_mask(ink_bits_db[gi], bitlen, size)
#         tv_z  = zones_tversky(q_mask, cmask, alpha=alpha, beta=beta)
#         m2, c2 = zone_stats_from_mask(cmask)
#         ot = sinkhorn_ot(q_mass, q_cent, m2, c2, reg=0.1, iters=50)
#         sims_tv[k]   = tv_z
#         sims_comp[k] = float(0.5*tv_z + 0.5*(1.0-ot))  # combine TV_z + (1-OT)

#     # normalize channels over candidates
#     chans = {
#         "hog": minmax01(sims_hog),
#         "tv" : minmax01(sims_tv),
#         "topo": minmax01(sims_top),
#         "comp": minmax01(sims_comp)
#     }

#     # per-class capped vote using primary order
#     order = np.argsort(-sims_primary)
#     class_scores = defaultdict(float); by_class=defaultdict(list)
#     for pos in order:
#         cls = labels[cand[pos]]
#         if len(by_class[cls])<cap: by_class[cls].append(pos)

#     for cls, lst in by_class.items():
#         sc = (w_hog*chans["hog"][lst].sum()
#               + w_tv*chans["tv"][lst].sum()
#               + w_topo*chans["topo"][lst].sum()
#               + w_comp*chans["comp"][lst].sum())
#         class_scores[cls]+=sc

#     # centroid backstop (HOG) for present classes
#     q_unit = q_hog_std/(np.linalg.norm(q_hog_std)+1e-12)
#     present = set(labels[cand])
#     for cls, cvec in cents.items():
#         if cls not in present: continue
#         s = float(np.dot(q_unit, cvec)/(np.linalg.norm(cvec)+1e-12))
#         class_scores[cls]+= max(0.0,s)*w_cent

#     ranked = sorted(class_scores.items(), key=lambda x:x[1], reverse=True)[:5]
#     conf = softmax([s for _,s in ranked], temperature=temp)*100.0

#     # representative example per class (highest HOG sim among candidates of that class)
#     results = []
#     for (cls, score), c in zip(ranked, conf):
#         # pick representative example from candidates by HOG similarity
#         best, bestv = None, -1.0
#         for i, gi in enumerate(cand):
#             if labels[gi] == cls and chans["hog"][i] > bestv:
#                 bestv = chans["hog"][i]
#                 best = gi

#         ex_path = files[best] if best is not None else None  # <-- define it

#         results.append({
#             "base": q_base,
#             "class": cls,
#             "similarity": float(score),
#             "confidence_percent": float(round(c, 2)),
#             "example": ex_path,  # <-- now defined
#         })



#     neighbors=[]
#     for i in order[:min(5, len(order))]:
#         gi = cand[i]
#     neighbors.append({
#         "base": bases[gi],
#         "class": labels[gi],
#         "file": files[gi],
#         "cosine_similarity": float(sims_primary[i]),
#         "cosine_distance": float(1.0 - sims_primary[i]),
#     })



#     return dict(
#         query=str(query_path), topk=5, neighbors_k=int(len(order)),
#         predictions=results, nearest_neighbors=neighbors
#     )

# # --------------------- CLI ---------------------
# def _default_paths():
#     return (
#         r"C:\Users\aravi\Downloads\quantum-kan\Img\akshara",
#         r"C:\Users\aravi\Downloads\quantum-kan\Img\.cache_akshara",
#         r"C:\Users\aravi\Downloads\quantum-kan\Img\akshara\ಕ\ಕೋ\img030-017.png",
#     )


# def _defaults():
#     # adjust if your paths differ
#     ds = Path(r"C:\Users\aravi\Downloads\quantum-kan\Img\akshara")
#     cache = Path(r"C:\Users\aravi\Downloads\quantum-kan\Img\.cache_akshara")
#     q = ds / "ಕ" / "ಕೋ" / "img030-017.png"
#     return ds, cache, q

# def main():
#     ap = argparse.ArgumentParser(description="Hybrid Kannada Akshara Search — v7 (unified CLI)")
#     ap.add_argument("--dataset", type=str, help="Dataset root (needed when building)")
#     ap.add_argument("--cache", type=str, required=True, help="Cache folder containing features.pkl (or to create one)")
#     ap.add_argument("--build", action="store_true", help="Build cache if missing")
#     ap.add_argument("--rebuild", action="store_true", help="Force rebuild cache")
#     ap.add_argument("--resize", type=int, default=192, help="Square resize dimension used during build")
#     ap.add_argument("--query", type=str, help="Full path to a query image")

#     # retrieval knobs
#     ap.add_argument("--neighbors", type=int, default=200, help="K candidates in coarse retrieval")
#     ap.add_argument("--cap", type=int, default=3, help="Per-class cap in fusion vote")

#     # fusion weights
#     ap.add_argument("--w_hog",  type=float, default=0.10)
#     ap.add_argument("--w_tv",   type=float, default=0.25)
#     ap.add_argument("--w_topo", type=float, default=0.20)
#     ap.add_argument("--w_comp", type=float, default=0.35)
#     ap.add_argument("--w_cent", type=float, default=0.05)
#     ap.add_argument("--temp",   type=float, default=0.25)
#     ap.add_argument("--alpha",  type=float, default=0.70)
#     ap.add_argument("--beta",   type=float, default=0.30)

#     # one-click convenience if no args
#     if len(sys.argv) == 1:
#         ds, cache, q = _defaults()
#         sys.argv += ["--cache", str(cache), "--query", str(q)]

#     args = ap.parse_args()
#     cache_dir = Path(args.cache); cache_dir.mkdir(parents=True, exist_ok=True)

#     if args.build or args.rebuild:
#         if not args.dataset:
#             print("[ERR] --dataset is required when building."); sys.exit(2)
#         ds = Path(args.dataset)
#         if not ds.exists():
#             print(f"[ERR] Dataset not found: {ds}"); sys.exit(2)
#         build_cache(ds, cache_dir, size=args.resize, force_rebuild=args.rebuild)

#     if args.query:
#         data = load_cache_and_indices(cache_dir)  # builds per-base NN indices
#         out = query_once(
#             data, Path(args.query),
#             neighbors_k=args.neighbors, cap=args.cap,
#             w_hog=args.w_hog, w_tv=args.w_tv, w_topo=args.w_topo, w_comp=args.w_comp, w_cent=args.w_cent,
#             temp=args.temp, alpha=args.alpha, beta=args.beta
#         )
#         print(json.dumps(out, indent=2, ensure_ascii=False))

# if __name__ == "__main__":
#     main()


# # def main():
# #     import argparse
# #     ap = argparse.ArgumentParser(description="Hybrid Kannada Akshara Search — v7 (pure sklearn)")
# #     ap.add_argument("--dataset", type=str, help="Dataset root (needed when building)")
# #     ap.add_argument("--cache", type=str, required=True, help="Cache folder containing features.pkl")
# #     ap.add_argument("--build", action="store_true")
# #     ap.add_argument("--rebuild", action="store_true")
# #     ap.add_argument("--resize", type=int, default=192)
# #     ap.add_argument("--query", type=str)

# #     ap.add_argument("--neighbors", type=int, default=200, help="K candidates in coarse retrieval")
# #     ap.add_argument("--cap", type=int, default=3, help="Per-class cap in fusion vote")

# #     ap.add_argument("--w_hog", type=float, default=0.10)
# #     ap.add_argument("--w_tv",  type=float, default=0.25)
# #     ap.add_argument("--w_topo",type=float, default=0.20)
# #     ap.add_argument("--w_comp",type=float, default=0.35)
# #     ap.add_argument("--w_cent",type=float, default=0.05)
# #     ap.add_argument("--temp",  type=float, default=0.25)
# #     ap.add_argument("--alpha", type=float, default=0.70)
# #     ap.add_argument("--beta",  type=float, default=0.30)

# #     if len(sys.argv)==1:
# #         ds, cache, q = _default_paths()
# #         sys.argv += ["--cache", cache, "--query", q]

# #     args = ap.parse_args()
# #     cache_dir = Path(args.cache)
# #     cache_dir.mkdir(parents=True, exist_ok=True)

# #     if args.build or args.rebuild:
# #         if not args.dataset:
# #             print("[ERR] --dataset is required when building."); sys.exit(2)
# #         ds = Path(args.dataset)
# #         if not ds.exists():
# #             print(f"[ERR] Dataset not found: {ds}"); sys.exit(2)
# #         build_cache(ds, cache_dir, size=args.resize, force_rebuild=args.rebuild)

# #     if args.query:
# #         data = load_cache_and_indices(cache_dir)
# #         out = query_once(
# #             data, Path(args.query),
# #             neighbors_k=args.neighbors, cap=args.cap,
# #             w_hog=args.w_hog, w_tv=args.w_tv, w_topo=args.w_topo, w_comp=args.w_comp, w_cent=args.w_cent,
# #             temp=args.temp, alpha=args.alpha, beta=args.beta
# #         )
# #         print(json.dumps(out, indent=2, ensure_ascii=False))

# # if __name__ == "__main__":
# #     main()




# -*- coding: utf-8 -*-
import json, re, time, warnings, tempfile, os, sys
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import cv2
from joblib import dump, load
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from skimage.feature import hog, local_binary_pattern
from skimage.morphology import skeletonize
from skimage.measure import euler_number

warnings.filterwarnings("ignore", category=UserWarning)

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ----------------------------
# Basic IO / dataset helpers
# ----------------------------
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    p = Path(path)
    try:
        data = p.read_bytes()
        arr = np.frombuffer(data, dtype=np.uint8)
        if arr.size:
            img = cv2.imdecode(arr, flags)
            if img is not None:
                return img
    except Exception:
        pass
    try:
        suffix = p.suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(p.read_bytes()); tmp_path = tmp.name
        img = cv2.imread(tmp_path, flags)
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        return img
    except Exception:
        return None

def is_image_file(p: Path) -> bool:
    try:
        return p.suffix.lower() in SUPPORTED_EXTS
    except Exception:
        return False

def list_class_image_paths(dataset_dir: Path):
    """Return list of (class_label, image_path) across nested or flat layout."""
    pairs = []
    subs = [p for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
    if subs:
        for base_dir in sorted(subs):
            for root, _, files in os.walk(base_dir):
                rp = Path(root)
                if any(part.startswith('.') for part in rp.parts): 
                    continue
                label = rp.name if rp != base_dir else base_dir.name
                for f in files:
                    p = rp / f
                    if is_image_file(p): 
                        pairs.append((label, str(p)))
        if pairs:
            return pairs

    # flat
    tok = re.compile(r'[_\-\s\.]+'); dig = re.compile(r'(\d+)'); word = re.compile(r'^[^\W\d_]+$', re.UNICODE)
    flats = [p for p in dataset_dir.iterdir() if p.is_file() and is_image_file(p)]
    if flats:
        def infer(stem):
            t = tok.split(stem)[0]
            if not t: return None
            t = dig.split(t)[0]
            return t if word.match(t) else None
        for p in flats:
            cls = infer(p.stem) or "all"
            pairs.append((cls, str(p)))
        return pairs
    raise RuntimeError(f"No images under: {dataset_dir}")

def base_from_path(dataset_dir: Path, file_path: str|Path) -> str:
    p = Path(file_path)
    try:
        rel = p.relative_to(dataset_dir)
        return rel.parts[0] if len(rel.parts) >= 1 else "all"
    except Exception:
        return p.parent.name or "all"

# ----------------------------
# Preprocess & features
# ----------------------------
def crop_to_ink(binary_img: np.ndarray, margin=3):
    mask = (binary_img < 128).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if ys.size == 0: 
        return binary_img
    y0 = max(int(ys.min())-margin, 0); y1 = min(int(ys.max())+1+margin, binary_img.shape[0])
    x0 = max(int(xs.min())-margin, 0); x1 = min(int(xs.max())+1+margin, binary_img.shape[1])
    return binary_img[y0:y1, x0:x1]

def preprocess(path: str, size=192) -> np.ndarray:
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        raise RuntimeError(f"read fail: {path}")
    if float(img.mean()) > 127: 
        img = 255 - img
    img = cv2.equalizeHist(img)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, 1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, 1)
    if th.mean() < 127: 
        th = 255 - th
    th = crop_to_ink(th, 3)
    h,w = th.shape[:2]; dim=max(h,w)
    pt=(dim-h)//2; pb=dim-h-pt; pl=(dim-w)//2; pr=dim-w-pl
    th = cv2.copyMakeBorder(th, pt,pb,pl,pr, cv2.BORDER_CONSTANT, 255)
    th = cv2.resize(th, (size,size), cv2.INTER_AREA)
    return th

# classic features
def hu_feature(img): 
    m=cv2.moments(img); hu=cv2.HuMoments(m).flatten()
    return (-np.sign(hu)*np.log10(np.abs(hu)+1e-12)).astype(np.float32)

def hog_feature(img):
    f = hog(img, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2),
            block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)
    return f.astype(np.float32)

def lbp_feature(img, P=8, R=2):
    lbp = local_binary_pattern(img, P=P, R=R, method="uniform")
    n_bins=P+2; edges=np.arange(0,n_bins+1,1,dtype=np.int32)
    hist,_ = np.histogram(lbp.ravel(), bins=edges, density=True)
    return hist.astype(np.float32)

def classical_vec(img): 
    return np.concatenate([hu_feature(img), hog_feature(img), lbp_feature(img)], 0)

# ----------------------------
# Bitpack & Tversky
# ----------------------------
def pack_mask(proc_img: np.ndarray) -> Tuple[np.ndarray, int]:
    mask = (proc_img < 128).astype(np.uint8).reshape(-1)
    return np.packbits(mask), int(mask.size)

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def tversky_bits(A: np.ndarray, B: np.ndarray, bitlen: int, alpha=0.7, beta=0.3) -> float:
    A = A.astype(np.uint8, copy=False); B = B.astype(np.uint8, copy=False)
    inter = int(_POP[np.bitwise_and(A,B)].sum())
    onlyA = int(_POP[np.bitwise_and(A, np.bitwise_not(B))].sum())
    onlyB = int(_POP[np.bitwise_and(B, np.bitwise_not(A))].sum())
    denom = inter + alpha*onlyA + beta*onlyB + 1e-12
    return float(inter/denom)

# ----------------------------
# Topology features (light)
# ----------------------------
def _neighbors8(y,x,H,W):
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy==0 and dx==0: 
                continue
            yy,xx=y+dy,x+dx
            if 0<=yy<H and 0<=xx<W: 
                yield yy,xx

def topo_features(proc_img: np.ndarray) -> np.ndarray:
    fg = (proc_img<128).astype(np.uint8)
    if fg.sum()==0:
        return np.zeros(16, dtype=np.float32)
    sk = skeletonize(fg>0).astype(np.uint8)
    H,W = sk.shape
    deg = np.zeros_like(sk, dtype=np.uint8)
    ys,xs = np.where(sk>0)
    for y,x in zip(ys,xs):
        c=0
        for yy,xx in _neighbors8(y,x,H,W):
            if sk[yy,xx]: c+=1
        deg[y,x]=c
    endpoints = int((deg==1).sum())
    junctions = int((deg>=3).sum())
    pathpx    = int((deg==2).sum())
    total     = int(sk.sum())
    loops     = int(max(0, -euler_number(sk, connectivity=2)))
    d_hist = np.array([ (deg==k).sum() for k in (0,1,2,3) ], dtype=np.float32)
    vec = np.concatenate([
        np.array([endpoints, junctions, pathpx, total, loops], dtype=np.float32),
        d_hist
    ], 0)
    if vec.sum()>0: vec = vec / (np.linalg.norm(vec)+1e-12)
    return vec.astype(np.float32)

# ----------------------------
# Softmax/minmax
# ----------------------------
def softmax(x, temperature=1.0):
    x=np.array(x,dtype=float); x/=float(temperature); x-=x.max()
    e=np.exp(x); return e/(e.sum()+1e-12)

def minmax01(a):
    a=np.asarray(a,dtype=float); mn, mx=a.min(), a.max()
    return np.zeros_like(a) if mx-mn<1e-12 else (a-mn)/(mx-mn)

# ----------------------------
# Build cache (pure sklearn)
# ----------------------------
def build_cache(dataset_dir: Path, cache_dir: Path, size=192, force_rebuild=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl = cache_dir/"features.pkl"
    if pkl.exists() and not force_rebuild:
        print(f"[INFO] Using existing cache: {pkl}"); 
        return

    pairs = list_class_image_paths(dataset_dir)
    labels=[]; files=[]; bases=[]
    X_hog=[]; X_topo=[]; ink_bits=[]
    t0=time.time(); print(f"[BUILD] Indexing {len(pairs)} from {dataset_dir}")
    for i,(cls,f) in enumerate(pairs,1):
        try:
            proc = preprocess(f, size=size)
            X_hog.append(classical_vec(proc))
            X_topo.append(topo_features(proc))
            pb, _ = pack_mask(proc); ink_bits.append(pb)
            labels.append(cls); files.append(f); bases.append(base_from_path(dataset_dir,f))
        except Exception as e:
            print(f"[WARN] skip {f}: {e}")
        if i%400==0: print(f"  {i}/{len(pairs)}")

    if not X_hog: 
        raise RuntimeError("no samples")

    X_hog = np.vstack(X_hog).astype(np.float32)
    X_topo= np.vstack(X_topo).astype(np.float32)
    labels=np.array(labels); files=np.array(files); bases=np.array(bases)
    ink_bits=np.array(ink_bits, dtype=object)

    scaler_hog = StandardScaler().fit(X_hog)
    scaler_top = StandardScaler(with_mean=False, with_std=True).fit(X_topo)
    X_hog_std = scaler_hog.transform(X_hog).astype(np.float32)
    X_top_std = scaler_top.transform(X_topo).astype(np.float32)

    # HOG class centroids
    centroids={}
    for cls in sorted(set(labels)):
        idx = np.where(labels==cls)[0]
        mu = X_hog_std[idx].mean(0); mu = mu/(np.linalg.norm(mu)+1e-12)
        centroids[cls]=mu.astype(np.float32)

    data = dict(
        X_hog_std=X_hog_std, X_top_std=X_top_std,
        scaler_hog=scaler_hog, scaler_top=scaler_top,
        labels=labels, files=files, bases=bases,
        ink_bits=ink_bits, size=int(size), bitlen=int(size*size),
        centroids_hog=centroids, dataset_dir=str(dataset_dir),
        built_at=time.ctime()
    )
    dump(data, pkl)
    print(f"[OK] Saved cache -> {pkl} (N={len(labels)})  in {time.time()-t0:.1f}s")

# ----------------------------
# Build per-base indices (sklearn)
# ----------------------------
def _build_per_base_indices(data: Dict[str,Any]):
    vecs = data["X_hog_std"]
    bases = data["bases"]
    base_to = {}
    uniq = sorted(set(bases))
    for b in uniq:
        idx = np.where(bases==b)[0]
        V = vecs[idx]
        nn = NearestNeighbors(metric='cosine', algorithm='brute')
        nn.fit(V)
        entry={"idx":idx, "backend":"sk", "index":nn, "vecs":V}
        base_to[b] = entry
    return base_to

def load_cache_and_indices(cache_dir: Path):
    data = load(cache_dir/"features.pkl")
    if "indices" not in data:
        data["indices"] = _build_per_base_indices(data)
        dump(data, cache_dir/"features.pkl")  # persist indices for faster next loads
    return data

# ----------------------------
# Query
# ----------------------------
def query_once(data: Dict[str,Any], query_path: Path,
               neighbors_k=200, cap=3,
               w_hog=0.40, w_tv=0.50, w_topo=0.05, w_comp=0.0, w_cent=0.05,
               temp=0.25, alpha=0.7, beta=0.3):
    size = int(data["size"]); bitlen=int(data["bitlen"])
    proc = preprocess(str(query_path), size=size)
    q_hog = classical_vec(proc)
    q_hog_std = data["scaler_hog"].transform(q_hog.reshape(1,-1)).astype(np.float32)[0]
    q_top = topo_features(proc)
    q_top_std = data["scaler_top"].transform(q_top.reshape(1,-1)).astype(np.float32)[0]
    q_bits, _ = pack_mask(proc)

    q_base = base_from_path(Path(data["dataset_dir"]), query_path)
    entry = data["indices"].get(q_base, None)
    if entry is None:
        entry = list(data["indices"].values())[0]
    idx_pool = entry["idx"]

    # coarse kNN (cosine)
    K = min(neighbors_k, len(idx_pool))
    dists, loc = entry["index"].kneighbors(q_hog_std.reshape(1,-1), n_neighbors=K)
    loc = loc[0]; sims_primary = 1.0 - dists[0]
    cand = idx_pool[loc]

    labels = data["labels"]; files = data["files"]; bases = data["bases"]
    X_hog_std = data["X_hog_std"]; X_top_std = data["X_top_std"]; ink_bits_db = data["ink_bits"]
    cents = data["centroids_hog"]

    # channels
    sims_hog = np.array([ float(np.dot(q_hog_std, X_hog_std[i]) /
                                ((np.linalg.norm(q_hog_std)+1e-12)*(np.linalg.norm(X_hog_std[i])+1e-12)))
                         for i in cand ], dtype=np.float32)
    sims_top = np.array([ float(np.dot(q_top_std, X_top_std[i]) /
                                ((np.linalg.norm(q_top_std)+1e-12)*(np.linalg.norm(X_top_std[i])+1e-12)))
                         for i in cand ], dtype=np.float32)
    sims_tv  = np.empty(len(cand), dtype=np.float32)
    for k, gi in enumerate(cand):
        sims_tv[k] = tversky_bits(q_bits, ink_bits_db[gi], bitlen, alpha=alpha, beta=beta)

    chans = {
        "hog": minmax01(sims_hog),
        "tv" : minmax01(sims_tv),
        "topo": minmax01(sims_top),
    }

    # per-class capped vote, ordered by primary
    order = np.argsort(-sims_primary)
    class_scores = defaultdict(float); by_class=defaultdict(list)
    for pos in order:
        cls = labels[cand[pos]]
        if len(by_class[cls])<cap: 
            by_class[cls].append(pos)
    for cls, lst in by_class.items():
        sc = (w_hog*chans["hog"][lst].sum()
              + w_tv*chans["tv"][lst].sum()
              + w_topo*chans["topo"][lst].sum())
        class_scores[cls]+=sc

    # centroid backstop (HOG)
    q_unit = q_hog_std/(np.linalg.norm(q_hog_std)+1e-12)
    present = set(labels[cand])
    for cls, cvec in cents.items():
        if cls not in present: 
            continue
        s = float(np.dot(q_unit, cvec)/(np.linalg.norm(cvec)+1e-12))
        class_scores[cls]+= max(0.0,s)*w_cent

    topk=5
    ranked = sorted(class_scores.items(), key=lambda x:x[1], reverse=True)[:topk]
    conf = softmax([s for _,s in ranked], temperature=temp)*100.0

        # build predictions
    results = []
    for (cls, score), c in zip(ranked, conf):
        best = None; bestv = -1
        for i, gi in enumerate(cand):
            if labels[gi] == cls and chans["hog"][i] > bestv:
                bestv = chans["hog"][i]; best = gi
        results.append({
            "base": q_base,
            "class": cls,  # <— renamed
            "similarity": float(score),
            "confidence_percent": float(round(c, 2)),
            "example": files[best] if best is not None else None
        })

    # neighbors preview
    neighbors = []
    for i in order[:min(topk, len(order))]:
        gi = cand[i]
        neighbors.append({
            "base": bases[gi],
            "class": labels[gi],  # <— renamed
            "file": files[gi],
            "cosine_similarity": float(sims_primary[i]),
            "cosine_distance": float(1.0 - sims_primary[i]),
        })

    return {
        "query": str(query_path),
        "topk": topk,
        "neighbors_k": int(neighbors_k),
        "predictions": results,
        "nearest_neighbors": neighbors,
    }

    # results=[]
    # for (cls,score),c in zip(ranked, conf):
    #     # pick repr example from candidates by hog sim
    #     best=None; bestv=-1
    #     for i,gi in enumerate(cand):
    #         if labels[gi]==cls and chans["hog"][i]>bestv:
    #             bestv=chans["hog"][i]; best=gi
    #     results.append(dict(
    #         base=q_base, clazz=cls, similarity=float(score),
    #         confidence_percent=float(round(c,2)),
    #         example=files[best] if best is not None else None
    #     ))

    # neighbors=[]
    # for i in order[:min(topk,len(order))]:
    #     gi=cand[i]
    #     neighbors.append(dict(
    #         base=bases[gi], clazz=labels[gi], file=files[gi],
    #         cosine_similarity=float(sims_primary[i]),
    #         cosine_distance=float(1.0-sims_primary[i])
    #     ))
    # return dict(
    #     query=str(query_path), topk=topk, neighbors_k=int(neighbors_k),
    #     predictions=results, nearest_neighbors=neighbors
    # )
