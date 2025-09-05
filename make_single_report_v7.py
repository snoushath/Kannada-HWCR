# make_single_report_v7.py
# Build ONE self-contained HTML report with metrics, confusion matrix image,
# and embedded error thumbnails (Top-1 wrong & GT-not-in-TopK).
import argparse, base64, csv, json, os, sys
from pathlib import Path
from datetime import datetime
from html import escape

MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

def data_uri(img_path: Path) -> str:
    ext = img_path.suffix.lower()
    mime = MIME.get(ext, "image/png")
    b = img_path.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"

def load_results(report_dir: Path):
    res_json = report_dir / "results.json"
    if res_json.exists():
        with open(res_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_predictions(report_dir: Path):
    rows = []
    csv_path = report_dir / "predictions.csv"
    if not csv_path.exists():
        return rows
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def find_error_thumbs(report_dir: Path, kind: str):
    """
    kind: 'top1_wrong' or 'not_in_topk'
    Returns list of (thumb_path, gt_class, filename) sorted by class then name.
    """
    root = report_dir / "errors" / kind
    out = []
    if not root.exists():
        return out
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for thumb in sorted(cls_dir.glob("*.png")):
            out.append((thumb, cls_dir.name, thumb.name))
    return out

def build_cards_html(section_title: str, examples, preds_idx, max_items=60):
    """
    examples: list of (thumb_path, gt_class, filename)
    preds_idx: dict query_path -> (pred_top, topk_list_str)
    """
    cards = []
    count = 0
    for thumb, gt, filename in examples:
        # Try to recover original path from the thumbnail naming convention:
        # e.g., img022-004_thumb.png -> img022-004.png
        original = filename.replace("_thumb", "")
        # Try to find a preds row by matching suffix
        pred_top, topk_s = "", ""
        for q, (pt, tk) in preds_idx.items():
            if q.endswith(original):
                pred_top, topk_s = pt, tk
                break
        meta_lines = []
        meta_lines.append(f"<div class='gt'>GT: {escape(gt)}</div>")
        if pred_top:
            meta_lines.append(f"<div class='pred'>Top-1: {escape(pred_top)}</div>")
        if topk_s:
            meta_lines.append(f"<div class='pred'>Top-k: {escape(topk_s)}</div>")
        src = data_uri(thumb)
        card = f"""
        <div class='card'>
          <img class='thumb' src='{src}'/>
          <div class='meta'>
            {''.join(meta_lines)}
            <div class='pred' style='color:#666'>…{escape(original)}</div>
          </div>
        </div>"""
        cards.append(card)
        count += 1
        if count >= max_items:
            break
    if not cards:
        return ""
    return f"""
    <h2>{escape(section_title)}</h2>
    <div class='grid'>
      {''.join(cards)}
    </div>
    """

def main():
    ap = argparse.ArgumentParser("Make ONE self-contained HTML report with embedded images")
    ap.add_argument("--report_dir", required=True, help="Folder that has results.json, predictions.csv, confusion_matrix.png, errors/")
    ap.add_argument("--out", default="report_single.html", help="Output HTML filename")
    ap.add_argument("--max_top1_wrong", type=int, default=60, help="Max examples to embed in Top-1 wrong section")
    ap.add_argument("--max_not_in_topk", type=int, default=60, help="Max examples for Not-in-TopK section")
    args = ap.parse_args()

    rep = Path(args.report_dir)
    out_html = rep / args.out

    res = load_results(rep)
    preds = load_predictions(rep)

    # index predictions for quick lookup
    preds_idx = {}
    for row in preds:
        q = row.get("query", "")
        pt = row.get("pred_top", "")
        tk = ""
        # If you later add a 'topk_list' col, use it. For now we reconstruct using rows if needed.
        # We leave empty if not available.
        preds_idx[q] = (pt, tk)

    # confusion matrix image (optional)
    cm_png = rep / "confusion_matrix.png"
    cm_html = ""
    if cm_png.exists():
        cm_src = data_uri(cm_png)
        cm_html = f"""
        <h2>Confusion Matrix</h2>
        <div class='imgwrap'>
          <img class='cm' src='{cm_src}'/>
        </div>
        """

    # headline metrics
    meta_html = ""
    if res:
        meta_html = f"""
        <div class='metrics'>
          <div><b>Samples</b>: {res.get('samples','')}</div>
          <div><b>Top-1</b>: {res.get('top1_percent','')}%</div>
          <div><b>Top-{res.get('k','')}</b>: {res.get('topk_percent','')}%</div>
          <div><b>mAP@{res.get('k','')}</b>: {res.get('map_at_k','')}%</div>
          <div><b>Split</b>: {escape(str(res.get('split','')))}</div>
        </div>
        """

    # error sections from thumbnails
    top1_wrongs = find_error_thumbs(rep, "top1_wrong")
    not_in_topk = find_error_thumbs(rep, "not_in_topk")

    top1_html = build_cards_html("Top-1 wrong predictions", top1_wrongs, preds_idx, max_items=args.max_top1_wrong)
    topk_html = build_cards_html("Ground truth not in Top-5", not_in_topk, preds_idx, max_items=args.max_not_in_topk)

    # assemble one HTML
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Akshara Retrieval — One-File Report</title>
<style>
body{{font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:24px; color:#111;}}
.header{{display:flex; align-items:baseline; gap:16px; margin-bottom:12px}}
h1{{font-size:24px; margin:0}}
.small{{color:#666; font-size:12px}}
.metrics{{display:grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap:8px; background:#fafafa; border:1px solid #eee; padding:12px; border-radius:8px; margin:8px 0 20px}}
.imgwrap{{text-align:center; margin:10px 0 24px}}
.cm{{max-width:100%; height:auto; border:1px solid #eee; border-radius:8px}}
.grid{{display:flex; flex-wrap:wrap; gap:12px}}
.card{{border:1px solid #ddd; border-radius:8px; padding:8px; width:210px}}
.thumb{{width:200px; height:auto; display:block; margin-bottom:6px}}
.meta{{font-size:12px; line-height:1.35}}
.gt{{font-weight:bold}}
h2{{margin-top:24px}}
hr{{border:none; border-top:1px solid #eee; margin:24px 0}}
</style>
</head><body>
<div class="header">
  <h1>Akshara Retrieval — Results Report</h1>
  <div class="small">generated: {escape(datetime.now().isoformat(sep=' ', timespec='seconds'))}</div>
</div>

{meta_html}
{cm_html}
{top1_html}
{topk_html}

<hr/>
<div class="small">This file is self-contained. Open locally or share directly. Print to PDF for submission.</div>
</body></html>
"""
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote {out_html}")

if __name__ == "__main__":
    main()
