import json
import numpy as np
import torch
import random
import os
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from tools import train, get_dataset, get_loaders, get_model, EarlyStopping
import argparse

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = "cpu"
EPOCHS = 2

g = torch.Generator()
g.manual_seed(SEED)

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Tiny demonstrator VERBOSEing how to configure the training pipeline."
    )

    parser.add_argument(
        "--mode",
        choices=["geometry", "geometry_plus"],
        default="geometry_plus",
        help=(
            "`geometry` – use only geometric coordinates (x, y, z) for each point (3 features); "
            "`geometry_plus` – adds the surface normal, giving 4 features (default)."
        ),
    )

    parser.add_argument(
        "-a", "--approach",
        choices=["Baseline", "Original", "K_means", "Quantilles"],
        default="K_means",
        help=(
            "`Baseline`   – no categorical edge labels (`K` is forced to 0).\n"
            "`Original`   – a single, fixed edge label (`K` is forced to 1).\n"
            "`K_means`    – automatic K‑means clustering with `K` categories (default).\n"
            "`Quantilles` – split edges into `K` quantile‑based bins."
        ),
    )

    parser.add_argument(
        "-k","--k",
        type=int,
        default=8,
        metavar="K",
        help=(
            "Number of edge categories. Ignored for `Baseline` and overridden to 1 when "
            "`Original` appraoch is selected."
        ),
    )

    parser.add_argument(
        "-ep", 
        "--epochs",
        type=int,
        default=200,
        help="Total number of training epochs (default: 200).",
    )

    parser.add_argument(
        "-bs",
        '--batch_size',
        type=int,
        default=256,
        metavar="BATCH_SIZE",
        help="Mini‑batch size.",
    )

    parser.add_argument(
        "-es",
        "--embedding_size",
        type=int,
        default=64,
        metavar="EMB_SIZE",
        help="Size of the latent embedding vector.",
    )

    parser.add_argument(
        "-d",
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help=(
            "`cpu`  – force CPU execution.\n"
            "`cuda` – use the first CUDA‑capable GPU (error if unavailable).\n"
            "`mps`  – Apple‑Silicon GPU via Metal Performance Shaders.\n"
            "`auto` – pick `cuda` if available, otherwise default to `cpu` (default)."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="visualize the results during training.",
        default=True
    )

    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="add '-t' to test the best model in Test Data.",
        default=True
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer (default: 1e-3).",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 penalty) for the optimizer (default: 1e-4).",
    )

    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Period of learning rate decay in epochs (default: 10).",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Multiplicative factor of learning rate decay (default: 0.95).",
)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    MODE = args.mode
    if MODE.endswith("plus"):
        data_type = "Geometry with physicochemical properties"
    else:
        data_type = "Geometry only"
    N_FEATURES = 4 if MODE == "geometry_plus" else 3

    K = args.k
    APPROACH = args.approach
    EDGE_DIM = 1 if APPROACH == "Baseline" else 3
    BATCH_SIZE = args.batch_size
    EMBEDDING_SIZE = args.embedding_size

    # Looking for the requested device
    requested_device = args.device
    if requested_device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but no compatible GPU detected – falling back to CPU.")
        DEVICE = "cpu"
    else:
        DEVICE = requested_device

    # Enforce approach‑specific constraints on K
    if APPROACH == "Baseline" and K != 0:
        print("`Baseline` approach selected – forcing number of categories (K) to 0.")
        K = 0
    elif APPROACH == "Original" and K != 1:
        print("`Original` approach selected – forcing number of categories (K) to 1.")
        K = 1

    LR_PARAMS = (1e-3, 1e-4, 10, 0.95)
    EPOCHS = 200 if args.epochs < 1 else args.epochs
    VERBOSE = args.verbose
    TEST = args.test

    RUN_ID = f"{MODE}_{APPROACH}_{K}"


    # Display the effective configuration
    print('-'*50)
    print(
        "\nEffective configuration:"
        f"\n  RUN ID          : {RUN_ID}"
        f"\n  DATA            : {data_type}"
        f"\n  APPROACH        : {APPROACH}"
        f"\n  K               : {K}"
        f"\n  BATCH_SIZE      : {BATCH_SIZE}"
        f"\n  EMBEDDING_SIZE  : {EMBEDDING_SIZE}"
        f"\n  EPOCHS          : {EPOCHS}"
        f"\n  DEVICE          : {DEVICE}"
        f"\n  VERBOSE         : {VERBOSE}"
        f"\n  TEST            : {TEST}"        
    )
    print('\n "python train.py --help" for mor informations \n')
    print('-'*50)

    dataset, labels, N_CLASSES = get_dataset(MODE, APPROACH, K)
    train_loader, val_loader, test_loader = get_loaders(dataset["train"], dataset["val"], dataset["test"], batch_size=BATCH_SIZE, seed=g)
    model, optimizer, scheduler, loss_fnc = get_model(EDGE_DIM, N_CLASSES, N_FEATURES, EMBEDDING_SIZE, LR_PARAMS)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    best_scores = train(RUN_ID, model, EDGE_DIM, train_loader, val_loader, test_loader, EPOCHS, DEVICE, N_CLASSES, loss_fnc, optimizer, scheduler, early_stopping, VERBOSE, TEST)
    best_scores = {k: float(v) for k, v in best_scores.items()}

    with open(f"results/{RUN_ID}.json", "w") as f:
        json.dump(best_scores, f, indent=4)
        print(f"Final scores were stored in : 'results/{RUN_ID}.json' ✅.")