#!/usr/bin/env python3
# examples/minimal_train.py
"""
Minimal training demo.

Example:
    python examples/minimal_train.py \
        --data_dir /path/to/dataset \
        --dataset dataset512 \
        --epochs 1 --batch_size 4 --lr 1e-3 --device auto --no-summary
"""

import os
import argparse
import random
import numpy as np
import torch

from tf_attn_yolo import TF_Attn_Yolo

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal training (no dummy inputs)")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root with train/ and val/ subfolders, each containing images/ and labels/.")
    p.add_argument("--dataset", type=str, default="dataset512",
                   choices=["multires", "unires"],
                   help="Which dataset loader to use in BaseModel.fit (default: dataset512).")
    p.add_argument("--epochs", type=int, default=1, help="Epochs (default: 1)")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--patience", type=int, default=3, help="Early stop patience (default: 3)")
    p.add_argument("--width_mult", type=float, default=0.25, help="Backbone width multiplier (default: 0.25)")
    p.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto', 'cpu', 'cuda:0', etc. (default: auto)")
    p.add_argument("--output_dir", type=str, default="outputs/minimal_train",
                   help="Where to save checkpoints & logs")
    return p.parse_args()


def pick_device(flag: str) -> str:
    if flag != "auto":
        return flag
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def assert_dataset_layout(root: str) -> None:
    expected = [
        os.path.join(root, "train", "data"),
        os.path.join(root, "train", "labels"),
        os.path.join(root, "val", "data"),
        os.path.join(root, "val", "labels"),
    ]
    missing = [p for p in expected if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(
            "Dataset layout is incomplete. Missing directories:\n- " + "\n- ".join(missing)
        )


def main() -> None:
    args = parse_args()
    set_seed(0)

    device = pick_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    assert_dataset_layout(args.data_dir)

    # --- Build model ---
    model = TF_Attn_Yolo(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        strides=[8, 16, 32],
        reg_max=16,
        device=device,
        input_canals=1,
        width_mult=args.width_mult,
    )

    # Optional labels for later plotting
    model.class_names = [f"class_{i}" for i in range(args.num_classes)]

    # --- Train purely from real data ---
    history = model.fit(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        dataset=args.dataset,    # <- choose your loader (no dummy tensors involved)
        use_amp=True,
        evaluator=None,
        monitor="val_loss",
        mode="min",
    )

    # --- Report ---
    if history:
        last = history[-1]
        print("\n=== Training done ===")
        print(f"Epochs run       : {last.get('epoch', len(history))}")
        print(f"Last train_loss  : {last.get('train_loss'):.4f}")
        print(f"Last val_loss    : {last.get('val_loss'):.4f}")
    else:
        print("\nTraining finished without recorded history (unexpected).")

    best_ckpt = os.path.join(args.output_dir, "best.pt")
    last_ckpt = os.path.join(args.output_dir, "last.pt")
    print(f"Best checkpoint  : {best_ckpt}  ({'exists' if os.path.isfile(best_ckpt) else 'missing'})")
    print(f"Last checkpoint  : {last_ckpt}  ({'exists' if os.path.isfile(last_ckpt) else 'missing'})")


if __name__ == "__main__":
    main()
