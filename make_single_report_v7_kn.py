# make_single_report_v7_kn.py
# One-file HTML report with Kannada axis labels in the confusion matrix.

import argparse, base64, csv, json
from pathlib import Path
from datetime import datetime
from html import escape
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt

# Try to prefer Kannada-capable fonts on Windows/Linux
PREFERRED_FONTS = [
    "Nirmala UI",           # Windows
    "Tunga",                # Windows
    "Noto Sans Kannada",    # cross-platform
    "Noto Serif Kannada",
    "Lohit Kannada",
    "Arial Unicode MS",     # broad coverage
    "Segoe UI Symbol",      # fallback
]

MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

def pick_kannada_font(user_font: str | None = None) -> str:
    from matplotlib.font_manager import fontManager
    names = {f.name for f in fontManager.ttflist}
    if user_font:
        return user_font
    for name in PREFERRED_FONTS:
        if name in names:
            return name
    # fallback to default
    return matplotlib.rcParams.get("font.sans-serif", ["sans-serif"])[0]

def data_uri_from_path(p: Path) -> str:
    ext = p.suffix.lower()
    mime = MIME.get(ext, "image/png")
    b = p.read_bytes()
    import base64
    return f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"

def data_uri_from_png_bytes(buf: bytes) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")

def load_results(report_dir: Path) -> dict:
    p = report_dir / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def load_predictions(report_dir: Path):
    rows = []
    p = report_dir / "predictions.csv"
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows

def find_error_thumbs(report_dir: Path, kind: str):
    """kind: 'top1_wrong' or 'not_in_topk'"""
    root = report_dir / "errors" / kind
    out = []
    if not root.exists():
        return out
    for cls_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        for thumb in sorted(cls_dir.glob("*.png")):
            out.append((thumb, cls_dir.name, thumb.name))
    return out

def build_cards_html(title: str, examples, preds_idx, max_items=60):
    cards, count = [], 0
    for thumb, gt, filename in examples:
        original = filename.replace("_thumb", "")  # our thumbnail naming
        pred_top, topk_s = "", ""
        # best-effort: match by suffix
        for q, (pt, tk) in preds_idx.items():
            if q.endswith(original):
                pred_top, topk_s = pt, tk
                break
        src = data_uri_from_path(thumb)
        card = f"""
        <div class='card'>
          <img class='thumb' src='{src}'/>
          <div class='meta'>
            <div class='gt'>GT: {escape(gt)}</div>
            <div class='pred'>Top-1: {escape(pred_top)}</div>
            {f"<div class='pred'>Top-k: {escape(topk_s)}</div>" if topk_s else ""}
            <div class='pred' style='color:#666'>…{escape(original)}</div>
          </div>
        </div>"""
        cards.append(card)
        count += 1
        if count >= max_items:
            break
    if not cards:
        return ""
    return f"<h2>{escape(title)}</h2><div class='grid'>{''.join(cards)}</div>"

def read_confusion_csv(conf_csv: Path):
    """Parse confusion_matrix.csv written by eval_v7.py.
       Expected: header row = ['', pred1, pred2, ...]
                 each next row = [gt, c11, c12, ...]"""
    if not conf_csv.exists():
        return None, None
    rows = []
    with conf_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        return None, None
    header = rows[0][1:]  # predicted class labels
    gt_labels = []
    mat = []
    for row in rows[1:]:
        if not row:
            continue
        gt_labels.append(row[0])
        mat.append([int(x) for x in row[1:]])
    import numpy as np
    return np.array(mat, dtype=int), (gt_labels, header)

def render_confusion_png_data(mat, labels, font_name: str, cmap="viridis") -> bytes:
    """Return PNG bytes for the confusion matrix figure with Kannada tick labels."""
    # Set font
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans", "Arial Unicode MS", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    im = ax.imshow(mat, cmap=cmap)
    ax.set_title("Confusion Matrix (counts)", pad=12)

    gt, pred = labels
    ax.set_xticks(range(len(pred)), labels=pred, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(gt)), labels=gt, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    # Optional value annotations for small matrices
    if mat.shape[0] <= 20:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if val:
                    ax.text(j, i, str(val), ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    bio = BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight")
    plt.close(fig)
    return bio.getvalue()

def main():
    ap = argparse.ArgumentParser("One-file report with Kannada CM axes")
    ap.add_argument("--report_dir", required=True, help="Folder with results.json, predictions.csv, confusion_matrix.csv, errors/")
    ap.add_argument("--out", default="report_single_kn.html", help="Output HTML filename")
    ap.add_argument("--font", default=None, help="Override font (e.g., 'Nirmala UI'). If omitted, best available is picked.")
    ap.add_argument("--max_top1_wrong", type=int, default=60)
    ap.add_argument("--max_not_in_topk", type=int, default=60)
    args = ap.parse_args()

    rep = Path(args.report_dir)
    out_html = rep / args.out

    # choose Kannada-capable font
    font_name = pick_kannada_font(args.font)

    # metrics + preds
    res = load_results(rep)
    preds = load_predictions(rep)
    preds_idx = {row.get("query",""): (row.get("pred_top",""), "") for row in preds}

    # error sections
    top1 = find_error_thumbs(rep, "top1_wrong")
    notk = find_error_thumbs(rep, "not_in_topk")

    # Confusion from CSV -> PNG bytes -> data URI
    cm_csv = rep / "confusion_matrix-new.csv"
    cm_html = ""
    mat, labels = read_confusion_csv(cm_csv)
    if mat is not None and labels is not None:
        png_bytes = render_confusion_png_data(mat, labels, font_name)
        cm_html = f"""
        <h2>Confusion Matrix</h2>
        <div class='imgwrap'><img class='cm' src='{data_uri_from_png_bytes(png_bytes)}'/></div>
        """
    else:
        # fallback to existing PNG if any
        cm_png = rep / "confusion_matrix.png"
        if cm_png.exists():
            cm_html = f"<h2>Confusion Matrix</h2><div class='imgwrap'><img class='cm' src='{data_uri_from_path(cm_png)}'/></div>"

    # Build sections
    meta_html = f"""
    <div class='metrics'>
      <div><b>Samples</b>: {res.get('samples','')}</div>
      <div><b>Top-1</b>: {res.get('top1_percent','')}%</div>
      <div><b>Top-{res.get('k','')}</b>: {res.get('topk_percent','')}%</div>
      <div><b>mAP@{res.get('k','')}</b>: {res.get('map_at_k','')}%</div>
      <div><b>Split</b>: {escape(str(res.get('split','')))}</div>
      <div><b>Font</b>: {escape(font_name)}</div>
    </div>
    """

    top1_html = build_cards_html("Top-1 wrong predictions", top1, preds_idx, args.max_top1_wrong)
    notk_html = build_cards_html("Ground truth not in Top-5", notk, preds_idx, args.max_not_in_topk)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Akshara Retrieval — Kannada Report</title>
<style>
body{{font-family:-apple-system, Segoe UI, "Nirmala UI", Roboto, Arial, sans-serif; margin:24px; color:#111}}
.header{{display:flex; align-items:baseline; gap:16px; margin-bottom:12px}}
h1{{font-size:24px; margin:0}}
.small{{color:#666; font-size:12px}}
.metrics{{display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:8px; background:#fafafa; border:1px solid #eee; padding:12px; border-radius:8px; margin:8px 0 20px}}
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
  <h1>Akshara Retrieval — Results (Kannada CM)</h1>
  <div class="small">generated: {escape(datetime.now().isoformat(sep=' ', timespec='seconds'))}</div>
</div>

{meta_html}
{cm_html}
{top1_html}
{notk_html}

<hr/>
<div class="small">This HTML is fully self-contained. You can open locally, share, or print to PDF for submission.</div>
</body></html>
"""
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote {out_html} using font: {font_name}")

if __name__ == "__main__":
    main()
