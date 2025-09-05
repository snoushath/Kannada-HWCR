# build_cache_oneclick.py
# One-click cache build for VS Code (no PowerShell needed)

from pathlib import Path
import sys

# >>> EDIT THESE TWO PATHS <<<
DATASET_DIR = Path(r"C:\Users\aravi\Downloads\quantum-kan\Img\akshara")
CACHE_DIR   = Path(r"C:\Users\aravi\Downloads\quantum-kan\Img\.cache_akshara")

RESIZE = 192       # square resize
REBUILD = True     # force rebuild cache

def main():
    # import the library and call the builder directly
    try:
        import akshara_lib_v7 as ak
    except Exception as e:
        print("[ERR] Could not import akshara_lib_v7.py. Make sure this file is in the same folder.")
        print(e)
        sys.exit(1)

    if not DATASET_DIR.exists():
        print(f"[ERR] Dataset folder not found: {DATASET_DIR}")
        sys.exit(2)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("[BUILD] starting…")
    ak.build_cache(DATASET_DIR, CACHE_DIR, size=RESIZE, force_rebuild=REBUILD)
    print("[BUILD] done.")

if __name__ == "__main__":
    main()
