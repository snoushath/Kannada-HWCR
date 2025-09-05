# akshara_eval_baseline_on_list.py
# Run the baseline (HOG/LBP/Hu + Tversky) on the SAME queries listed in predictions-pr.csv

from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from akshara_lib_compat import load_cache_and_indices, query_once

def main():
    ap = argparse.ArgumentParser("Baseline eval on a fixed query list")
    ap.add_argument("--cache", required=True, help="Cache dir with features.pkl")
    ap.add_argument("--queries_csv", required=True, help="CSV with columns: query, gt (use predictions-pr.csv from Ours run)")
    ap.add_argument("--out_csv", required=True, help="Where to write baseline predictions CSV")
    ap.add_argument("--neighbors", type=int, default=200)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    # (optional) tweak baseline channel weights if you want
    ap.add_argument("--w_hog", type=float, default=0.50)
    ap.add_argument("--w_tv",  type=float, default=0.50)
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    data = load_cache_and_indices(cache_dir)

    dfq = pd.read_csv(args.queries_csv)
    if not {"query", "gt"}.issubset(dfq.columns):
        raise SystemExit(f"{args.queries_csv} must have columns: query, gt")

    rows_out = []
    hits1 = hitsk = 0
    ap_sum = 0.0
    K = int(args.k)

    def topk_list(preds): return [p["class"] for p in preds[:K]]
    def hit_at_k(preds, gt): return gt in topk_list(preds)
    def ap_at_k(preds, gt):
        lst = topk_list(preds)
        try: i = lst.index(gt); return 100.0/(i+1)
        except ValueError: return 0.0

    for i, r in enumerate(dfq.itertuples(index=False), 1):
        if i % 50 == 0: print(f"  [{i}/{len(dfq)}] … {Path(r.query).name}")
        out = query_once(
            data, Path(r.query), neighbors_k=args.neighbors, cap=args.cap,
            w_hog=args.w_hog, w_tv=args.w_tv, w_topo=0.0, w_comp=0.0, w_cent=0.0
        )
        preds = out["predictions"]
        top1 = preds[0]["class"] if preds else ""
        rows_out.append({
            "query": r.query,
            "gt": r.gt,
            "pred_top": top1,
            "topk_list": ";".join(topk_list(preds)),
        })
        if hit_at_k(preds, r.gt): hitsk += 1
        if top1 == r.gt: hits1 += 1
        ap_sum += ap_at_k(preds, r.gt)

    # write CSV
    out_csv = Path(args.out_csv)
    pd.DataFrame(rows_out).to_csv(out_csv, index=False, encoding="utf-8")
    N = len(rows_out)
    summary = {
        "N": N,
        "Hit@1_%": round(100.0*hits1/N, 2) if N else 0.0,
        "Hit@5_%": round(100.0*hitsk/N, 2) if N else 0.0,
        "mAP@5_%": round(ap_sum/N, 2) if N else 0.0,
        "k": K
    }
    print("\n[Baseline] Summary:", json.dumps(summary, indent=2))
    print(f"[OUT] {out_csv}")

if __name__ == "__main__":
    main()
