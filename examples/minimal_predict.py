#!/usr/bin/env python3
# examples/minimal_predict.py
"""
Minimal inference: load ONE tensor from val/images and run predict.

Example:
    python examples/minimal_predict.py \
        --data_dir /path/to/dataset \
        --weights outputs/minimal_train/best.pt \
        --device auto \
        --save_plot
"""

import os
import argparse
import numpy as np
import torch
from ..tf_attn_yolo import TF_Attn_Yolo

def parse_args():
    p = argparse.ArgumentParser(description="Minimal single-sample predict")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Dataset root with val/images/ and val/labels/ (labels unused here).")
    p.add_argument("--weights", type=str, default="",
                   help="Optional path to .pt/.pth weights.")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto', 'cpu', 'cuda:0', etc.")
    p.add_argument("--num_classes", type=int, default=3,
                   help="Must match training.")
    p.add_argument("--width_mult", type=float, default=0.25,
                   help="Keep consistent with training.")
    p.add_argument("--conf_thres", type=float, default=0.1)
    p.add_argument("--output_dir", type=str, default="outputs/minimal_predict")
    p.add_argument("--save_plot", action="store_true",
                   help="If set, saves a visualization with predicted boxes.")
    return p.parse_args()


def pick_device(flag: str) -> str:
    if flag != "auto":
        return flag
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def find_first_sample(images_dir: str) -> str:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Missing directory: {images_dir}")
    cand = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".pt", ".npy"))])
    if not cand:
        raise FileNotFoundError(f"No .pt or .npy files found in {images_dir}")
    return os.path.join(images_dir, cand[0])


def load_tensor_any(path: str):
    """
    Returns either:
      - torch.Tensor of shape (C,H,W) or (1,C,H,W)
      - list[torch.Tensor] for multi-res (each (C,H,W) or (1,C,H,W))
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        obj = torch.load(path, map_location="cpu")
    elif ext == ".npy":
        arr = np.load(path)
        obj = torch.from_numpy(arr)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    # Multi-resolution case stored as list/dict
    if isinstance(obj, dict):
        # keep deterministic order by key
        obj = [obj[k] for k in sorted(obj.keys())]
    if isinstance(obj, list):
        tens_list = []
        for t in obj:
            t = torch.as_tensor(t)
            if t.dim() == 2:  # (H,W) -> (1,H,W)
                t = t.unsqueeze(0)
            elif t.dim() == 3:  # (C,H,W)
                pass
            else:
                raise RuntimeError(f"Unexpected tensor shape in list: {tuple(t.shape)}")
            # add batch dim: (1,C,H,W)
            t = t.unsqueeze(0)
            tens_list.append(t)
        return tens_list

    # Single tensor
    t = torch.as_tensor(obj)
    if t.dim() == 2:          # (H,W) -> (1,H,W)
        t = t.unsqueeze(0)
    elif t.dim() == 3:        # (C,H,W)
        pass
    else:
        raise RuntimeError(f"Unexpected tensor shape: {tuple(t.shape)}")
    t = t.unsqueeze(0)        # (1,C,H,W)
    return t


def main():
    args = parse_args()
    device = pick_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    images_dir = os.path.join(args.data_dir, "val", "images")
    sample_path = find_first_sample(images_dir)
    print(f"[info] Using sample: {sample_path}")

    # Load sample to CPU first; model.predict will move it to device
    sample = load_tensor_any(sample_path)

    # Build model
    model = TF_Attn_Yolo(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        strides=[8, 16, 32],
        reg_max=16,
        device=device,
        input_canals=1,
        width_mult=args.width_mult,
    )
    model.class_names = [f"class_{i}" for i in range(args.num_classes)]
    model.eval()

    # Load weights (optional)
    if args.weights:
        if not os.path.isfile(args.weights):
            raise FileNotFoundError(f"Weights not found: {args.weights}")
        missing, unexpected = model.load_weights(args.weights, device=device, eval_mode=True)
        if missing:
            print(f"[warn] Missing keys: {len(missing)} (first 5): {missing[:5]}")
        if unexpected:
            print(f"[warn] Unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")

    # Choose save path (optional)
    save_path = None
    if args.save_plot:
        base = os.path.splitext(os.path.basename(sample_path))[0]
        save_path = os.path.join(args.output_dir, f"{base}_pred.png")

    # Run predict (no GT labels here)
    preds, _, _ = model.predict(
        image_tensor=sample,
        to_plot=save_path if args.save_plot else False,
        conf_threshold=args.conf_thres,
        labels=None,
    )

    # Print quick summary
    n_img = len(preds)
    counts = [0 if p is None else int(p.shape[0]) for p in preds]
    print("\n=== Minimal predict ===")
    print(f"Device         : {device}")
    print(f"Images         : {n_img}")
    print(f"Detections/img : {counts}")
    if save_path:
        print(f"Saved plot     : {save_path}  ({'ok' if os.path.isfile(save_path) else 'failed'})")


if __name__ == "__main__":
    main()
