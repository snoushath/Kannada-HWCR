# # -*- coding: utf-8 -*-
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
# from skimage.measure import label as cc_label, regionprops, euler_number


# def _defaults():
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
#     ap.add_argument("--resize", type=int, default=192, help="Square resize used during build")
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


# -*- coding: utf-8 -*-
import argparse, json, sys
from pathlib import Path
from akshara_lib_v7 import build_cache, load_cache_and_indices, query_once

def _defaults():
    ds = Path(r"C:\Users\aravi\Downloads\quantum-kan\Img\akshara")
    cache = Path(r"C:\Users\aravi\Downloads\quantum-kan\Img\.cache_akshara")
    q = ds / "ಕ" / "ಕೋ" / "img030-017.png"
    return ds, cache, q

def main():
    ap = argparse.ArgumentParser(description="Hybrid Kannada Akshara Search — v7 (CLI)")
    ap.add_argument("--dataset", type=str, help="Dataset root (needed when building)")
    ap.add_argument("--cache", type=str, required=True, help="Cache folder containing features.pkl (or to create one)")
    ap.add_argument("--build", action="store_true", help="Build cache if missing")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild cache")
    ap.add_argument("--resize", type=int, default=192, help="Square resize used during build")
    ap.add_argument("--query", type=str, help="Full path to a query image")

    ap.add_argument("--neighbors", type=int, default=200, help="K candidates in coarse retrieval")
    ap.add_argument("--cap", type=int, default=3, help="Per-class cap in fusion vote")
    ap.add_argument("--w_hog",  type=float, default=0.40)
    ap.add_argument("--w_tv",   type=float, default=0.50)
    ap.add_argument("--w_topo", type=float, default=0.05)
    ap.add_argument("--w_cent", type=float, default=0.05)
    ap.add_argument("--temp",   type=float, default=0.25)
    ap.add_argument("--alpha",  type=float, default=0.70)
    ap.add_argument("--beta",   type=float, default=0.30)

    if len(sys.argv)==1:
        ds, cache, q = _defaults()
        sys.argv += ["--cache", str(cache), "--query", str(q)]

    args = ap.parse_args()
    cache_dir = Path(args.cache); cache_dir.mkdir(parents=True, exist_ok=True)

    if args.build or args.rebuild:
        if not args.dataset:
            print("[ERR] --dataset is required when building."); sys.exit(2)
        ds = Path(args.dataset)
        if not ds.exists():
            print(f"[ERR] Dataset not found: {ds}"); sys.exit(2)
        build_cache(ds, cache_dir, size=args.resize, force_rebuild=args.rebuild)

    if args.query:
        data = load_cache_and_indices(cache_dir)
        out = query_once(
            data, Path(args.query),
            neighbors_k=args.neighbors, cap=args.cap,
            w_hog=args.w_hog, w_tv=args.w_tv, w_topo=args.w_topo, w_cent=args.w_cent,
            temp=args.temp, alpha=args.alpha, beta=args.beta
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
