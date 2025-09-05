# check_cache_v7.py
import argparse, json
from pathlib import Path
from collections import Counter

def main():
    ap = argparse.ArgumentParser("Quick cache sanity-check")
    ap.add_argument("--cache", required=True, help="Folder that contains features.pkl")
    ap.add_argument("--query", help="Optional image to test a single retrieval")
    args = ap.parse_args()

    # Import here so it uses *your* akshara_lib_v7.py
    from akshara_lib_v7 import load_cache_and_indices, query_once

    cache_dir = Path(args.cache)
    if not (cache_dir / "features.pkl").exists():
        print(f"[ERR] No features.pkl under {cache_dir}")
        return

    # Be tolerant to old/new signatures
    try:
        data = load_cache_and_indices(cache_dir, backend="sk")
    except TypeError:
        data = load_cache_and_indices(cache_dir)

    files  = data.get("files", [])
    labels = data.get("labels", [])
    bases  = data.get("bases", [])

    # Basic summary
    info = {
        "N_samples": len(files),
        "N_classes": len(set(labels)),
        "size_cached": int(data.get("size", -1)),
        "has_keys": sorted(list(data.keys())),
        "X_hog_std_shape": tuple(data.get("X_hog_std", []).shape) if hasattr(data.get("X_hog_std", []), "shape") else None,
        "X_top_std_shape": tuple(data.get("X_top_std", []).shape) if hasattr(data.get("X_top_std", []), "shape") else None,
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))

    # Per-base counts
    cnt = Counter(bases)
    print("[BASE COUNTS]", dict(cnt))

    # Optional single retrieval smoke test
    if args.query:
        out = query_once(
            data, Path(args.query),
            neighbors_k=50, cap=3,
            w_hog=0.10, w_tv=0.25, w_topo=0.20, w_comp=0.35, w_cent=0.05,
            temp=0.25, alpha=0.7, beta=0.3
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
