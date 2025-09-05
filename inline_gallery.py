# inline_gallery.py
# Make your Akshara HTML reports mobile-safe.
#   inline (default): inline <img> files as base64 → single-file HTML
#   fix: only replace backslashes in src/href with forward slashes
#
# Usage (PowerShell / VS Code Terminal):
#   python inline_gallery.py --html "C:\...\reports-pr\error_gallery-pr.html"
#   python inline_gallery.py --html "C:\...\reports-pr\report_single-pr.html"
#   python inline_gallery.py --html "C:\...\report_single-pr.html" --mode fix

import argparse, re, base64, mimetypes
from pathlib import Path
from urllib.parse import unquote

IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}

def _posix(s: str) -> str:
    return s.replace("\\", "/")

def _data_uri(p: Path) -> str | None:
    try:
        data = p.read_bytes()
    except Exception:
        return None
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        ext = p.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        }.get(ext, "application/octet-stream")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"

def make_fixed(html_path: Path, out_path: Path) -> None:
    s = html_path.read_text(encoding="utf-8", errors="ignore")
    def repl(m):
        attr  = m.group(1)          # src|href
        quote = m.group(2)          # " or '
        val   = m.group(3).replace("\\", "/")
        return f'{attr}={quote}{val}{quote}'
    s = re.sub(r'(src|href)=(["\'])([^"\']+)\2', repl, s, flags=re.IGNORECASE)
    out_path.write_text(s, encoding="utf-8")
    print(f"[OK] Fixed paths → {out_path}")

def make_inline(html_path: Path, out_path: Path) -> None:
    root = html_path.parent
    s = html_path.read_text(encoding="utf-8", errors="ignore")

    # First normalize slashes inside src/href to help resolution
    def fix_attr(m):
        attr  = m.group(1); quote = m.group(2)
        val   = m.group(3).replace("\\", "/")
        return f'{attr}={quote}{val}{quote}'
    s = re.sub(r'(src|href)=(["\'])([^"\']+)\2', fix_attr, s, flags=re.IGNORECASE)

    # Inline <img ... src="...">
    def repl_img(m):
        whole = m.group(0)
        quote = m.group(2)
        src   = m.group(3)
        url = _posix(unquote(src))
        if url.startswith(("data:", "http://", "https://")):
            return whole
        rel = Path(url)
        candidates = [root / rel, root / rel.name]  # same folder fallback
        for p in candidates:
            if p.exists() and p.suffix.lower() in IMG_EXTS:
                du = _data_uri(p)
                if du:
                    return whole.replace(f"src={quote}{src}{quote}", f"src={quote}{du}{quote}")
        return whole

    s = re.sub(r'(<img\b[^>]*\ssrc=(["\'])([^"\']+)\2[^>]*>)', repl_img, s, flags=re.IGNORECASE)
    out_path.write_text(s, encoding="utf-8")
    print(f"[OK] Wrote single-file HTML → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inline or fix assets in Akshara report HTML.")
    ap.add_argument("--html", required=True, help="Path to error_gallery-pr.html or report_single-pr.html")
    ap.add_argument("--mode", choices=["inline", "fix"], default="inline")
    ap.add_argument("--out", default=None, help="Optional output path")
    args = ap.parse_args()

    html = Path(args.html)
    if not html.exists():
        raise SystemExit(f"[ERR] Not found: {html}")

    if args.out:
        out = Path(args.out)
    else:
        suffix = "-inline.html" if args.mode == "inline" else "-fixed.html"
        out = html.with_name(html.stem + suffix)

    if args.mode == "inline":
        make_inline(html, out)
    else:
        make_fixed(html, out)
