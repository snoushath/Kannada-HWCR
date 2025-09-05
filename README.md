**Akshara Hybrid Retrieval (AHR) — Lightweight, Script-Agnostic Character Retrieval for Indic Scripts**

AHR is a training-free, CPU-friendly retrieval pipeline for handwritten aksharas (here: Kannada; easily extended to other Indic scripts). It fuses appearance (HOG/LBP/Hu), topology (skeleton statistics), and zonal composition (asymmetric Tversky + centroid Sinkhorn-OT). Search is coarse-to-fine: fast per-base cosine k-NN first, then heavier channels on the top-K neighbors with per-class capping and weighted fusion.
