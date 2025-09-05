# build_cache_v7.py
from pathlib import Path
import argparse
from akshara_lib_v7 import build_cache, load_cache_and_indices

def main():
    ap = argparse.ArgumentParser("Build akshara cache (per-base indices)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--resize", type=int, default=192)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--ann", choices=["auto","faiss","hnsw","sk"], default="auto")
    args = ap.parse_args()

    ds = Path(args.dataset); cache = Path(args.cache)
    build_cache(ds, cache, size=args.resize, force_rebuild=args.rebuild)
    load_cache_and_indices(cache, backend=args.ann)  # builds ANN
    print("[BUILD] done.")

if __name__ == "__main__":
    main()
