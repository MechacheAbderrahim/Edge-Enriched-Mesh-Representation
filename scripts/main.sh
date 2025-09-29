#!/bin/bash

# Explanation of parameters:
# --mode: "geometry" → only 3D coordinates (x, y, z). "geometry_plus" → adds electrostatic property to coordinates.
# -a, --approach: edge categorization strategy {"Baseline", "Original", "K_means", "Quantilles"}.
# -k, --k: Number of edge categories. Ignored for "Baseline" and overridden to 1 when "Original" approach is selected.
# -ep, --epochs: Total number of training epochs (default: 200).
# -bs, --batch_size: Mini-batch size.
# -es, --embedding_size: Size of the latent embedding vector.
# -d, --device: 
#   "cpu"  – force CPU execution.
#   "cuda" – use the first CUDA-capable GPU (error if unavailable).
#   "mps"  – Apple-Silicon GPU via Metal Performance Shaders.
#   "auto" – pick "cuda" if available, otherwise default to "cpu" (default).
# --lr: Learning rate (default 1e-3).
# --weight_decay: Weight decay (L2 penalty, default 1e-4).
# --step_size: Scheduler step size in epochs (default 10).
# --gamma: Scheduler multiplicative decay factor (default 0.95).
# --verbose (flag) : visualize training (default ON)
# --test (flag) : evaluate best model on test data (default ON)

python3 main.py
    --mode geometry_plus \
    --approach K_means \
    --k 8 \
    --epochs 200 \
    --batch_size 256 \
    --embedding_size 64 \
    --device auto \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --step_size 10 \
    --gamma 0.95 \
    --verbose \
    --t
    