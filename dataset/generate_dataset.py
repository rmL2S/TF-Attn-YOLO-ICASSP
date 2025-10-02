from __future__ import annotations
from typing import Dict, Iterable, List, Literal, Optional, Tuple
import json
from pathlib import Path
from tqdm import tqdm

import torch

from .signals import chirp as synth_chirp, bpsk_code as synth_bpsk, impulses as synth_impulses
from .signals import band_noise as synth_band_noise, set_snr_from_noise
from .stft import stft_torch

SignalKind = Literal["chirp", "bpsk_code", "impulses"]

def _p(d: dict, *keys, default=None, required=True):
    """
    Return first existing key from *keys in dict d.
    If required and none found, raise a KeyError listing the candidate keys.
    """
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    if required:
        missing = " / ".join(keys)
        raise KeyError(f"Missing required param(s): {missing}")
    return default

def _normalize_params(kind: str, params: Dict) -> Dict:
    """
    Normalize scenario params into what the synth functions expect.
    Handles legacy/new keys (backward-compatible).
    """
    p = dict(params)

    # Ensure fs is float
    if "fs" in p:
        p["fs"] = float(p["fs"])

    if kind == "chirp":
        # synth_chirp expects: f0, f1, phase0 (radians), amplitude, fs
        p["f0"] = float(_p(p, "f0", "f0_hz"))
        p["f1"] = float(_p(p, "f1", "f1_hz"))
        # phase optional (support phase0_rad alias)
        p["phase0"] = float(_p(p, "phase0", "phase0_rad", required=False, default=0.0))
        # Cleanup aliases to avoid unexpected kwargs
        for k in ("f0_hz", "f1_hz", "phase0_rad"):
            p.pop(k, None)

    elif kind == "bpsk_code":
        # synth_bpsk expects: chip_rate, carrier_hz, phase0, amplitude, fs
        p["chip_rate"] = float(_p(p, "chip_rate", "chip_rate_hz"))
        p["carrier_hz"] = float(_p(p, "carrier_hz", "fc_hz"))
        p["phase0"] = float(_p(p, "phase0", "phase0_rad", required=False, default=0.0))
        for k in ("chip_rate_hz", "fc_hz", "phase0_rad"):
            p.pop(k, None)

    elif kind == "impulses":
        # Assuming synth_impulses expects: prf_hz, duty_cycle, width_s, edge_taper_samples, amplitude, fs
        # (si ton implé diffère, map ici les alias nécessaires)
        p["prf_hz"] = float(_p(p, "prf_hz"))
        p["duty_cycle"] = float(_p(p, "duty_cycle"))
        p["width_s"] = float(_p(p, "width_s"))
        p["edge_taper_samples"] = int(_p(p, "edge_taper_samples", required=False, default=0))
        p["amplitude"] = float(_p(p, "amplitude", required=False, default=1.0))

    return p


@torch.no_grad()
def generate_dataset(
    scenarios_path: str,
    out_dir: str,
    stft_configs: Iterable[Dict],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    shuffle_seed: Optional[int] = 123,
    log_magnitude: bool = True,
) -> None:
    """
    Build a multi-STFT spectrogram dataset from a JSONL scenarios file.

    Output layout:
        out_dir/{train,val,test}/data/*.pt    # list[Tensor] of spectrograms
        out_dir/{train,val,test}/labels/*.txt # Ultralytics format per sample
    """
    device = torch.device(device)
    out = Path(out_dir)
    for split in ("train", "val", "test"):
        (out / split / "data").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    scenarios: List[Dict] = []
    with open(scenarios_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))

    idxs = list(range(len(scenarios)))
    if shuffle_seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(shuffle_seed))
        perm = torch.randperm(len(idxs), generator=g).tolist()
        idxs = [idxs[i] for i in perm]

    n_total = len(idxs)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    splits = {
        "train": idxs[:n_train],
        "val": idxs[n_train:n_train + n_val],
        "test": idxs[n_train + n_val:],
    }

    for split, split_idxs in splits.items():
        for si in tqdm(split_idxs, desc=f"Processing {split}", unit="scenario"):
            scn = scenarios[si]
            base = f"{si:06d}"
            data_path = out / split / "data" / f"{base}.pt"
            label_path = out / split / "labels" / f"{base}.txt"

            spec_list = _synthesize_and_spectrograms(
                scn, stft_configs, device=device, dtype=dtype, log_magnitude=log_magnitude
            )
            torch.save(spec_list, data_path)

            yolo_lines = _make_ultralytics_labels(scn)
            with open(label_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(yolo_lines))


@torch.no_grad()
def _synthesize_and_spectrograms(
    scn: Dict,
    stft_configs: Iterable[Dict],
    *,
    device: torch.device,
    dtype: torch.dtype,
    log_magnitude: bool,
) -> List[torch.Tensor]:
    fs = float(scn["fs"])
    n = int(scn["n_samples"])

    # 1) Bruit de fond global (présent partout)
    bg_std = 1.0
    bg = torch.randn(n, dtype=torch.float32, device=device) * bg_std
    bg = bg.to(dtype)

    # 2) Interférence stationnaire (optionnelle) — **INR** vs bg
    inter_total = torch.zeros(n, dtype=dtype, device=device)
    if scn.get("interference") and scn["interference"] is not None and scn["interference"].get("present", False):
        inter = scn["interference"]
        bands = inter["bands_hz"]
        edge_taper_bins = int(inter.get("edge_taper_bins", 0))
        i, _ = synth_band_noise(
            n,
            fs=fs,
            bands_hz=[(float(a), float(b)) for (a, b) in bands],
            std=1.0,
            edge_taper_bins=edge_taper_bins,
            dtype=dtype,
            device=device,
        )
        # Compat: accepte 'inr_db' (nouveau) ou 'sir_db' (vieux json)
        inr_db = float(inter.get("inr_db", inter.get("sir_db", 0.0)))
        i = set_snr_from_noise(i, bg, inr_db)   # INR = SNR vs bruit
        inter_total += i

    # 3) Somme des signaux **scalés** pour respecter leur SNR vs bruit de fond
    sig_sum = torch.zeros(n, dtype=dtype, device=device)
    for s in scn["signals"]:
        kind: SignalKind = s["kind"]
        start = int(s["start_sample"])
        dur = int(s["duration_samples"])
        end = min(n, start + dur)
        if end <= start:
            continue

        local_n = end - start
        params = dict(s["params"])
        params["fs"] = float(params["fs"])  # ensure float
        nparams = _normalize_params(kind, params)

        # synthèse locale
        if kind == "chirp":
            x, _ = synth_chirp(local_n, **nparams, dtype=dtype, device=device)
        elif kind == "bpsk_code":
            x, _ = synth_bpsk(local_n, **nparams, dtype=dtype, device=device)
        elif kind == "impulses":
            x, _ = synth_impulses(local_n, **nparams, dtype=dtype, device=device)
        else:
            continue

        # SNR désiré vs **bruit de fond** sur la même fenêtre
        snr_db = float(s["snr_db"])
        bg_win = bg[start:end]
        x = set_snr_from_noise(x, bg[start:end], float(s["snr_db"]))
        sig_sum[start:end] += x

    # 4) Signal final = bruit de fond + interférence + signaux
    sig = bg + inter_total + sig_sum

    # 5) Multi-STFT export
    specs: List[torch.Tensor] = []
    for cfg in stft_configs:
        f, t, S = stft_torch(signal=sig, fs=fs, **cfg, device=device)
        A = torch.abs(S).to(dtype)
        specs.append(torch.log1p(A) if log_magnitude else A)

    return specs



@torch.no_grad()
def _make_ultralytics_labels(scn: Dict) -> List[str]:
    """
    Build YOLO labels (class xc yc w h) in relative time-frequency box space.
    Accepts legacy/new param names for backward compatibility.
    """
    fs = float(scn["fs"])
    T_total = float(scn["duration_s"])
    F_total = fs / 2.0

    def to_rel_box(t0: float, t1: float, f0: float, f1: float) -> Tuple[float, float, float, float]:
        t0, t1 = max(0.0, min(T_total, t0)), max(0.0, min(T_total, t1))
        if t1 < t0:
            t0, t1 = t1, t0
        f0, f1 = max(0.0, min(F_total, f0)), max(0.0, min(F_total, f1))
        if f1 < f0:
            f0, f1 = f1, f0
        x_c = ((t0 + t1) * 0.5) / T_total
        y_c = ((f0 + f1) * 0.5) / F_total
        w = max(1e-9, (t1 - t0) / T_total)
        h = max(1e-9, (f1 - f0) / F_total)
        return x_c, y_c, w, h

    def cls_id(kind: str) -> int:
        if kind == "impulses":
            return 0
        if kind == "bpsk_code":
            return 1
        if kind == "chirp":
            return 2
        return 2

    lines: List[str] = []
    for s in scn["signals"]:
        kind = s["kind"]
        t0 = float(s["start_sample"]) / fs
        t1 = t0 + float(s["duration_samples"]) / fs

        if kind == "chirp":
            p = s["params"]
            # Required keys
            f0 = float(_p(p, "f0_hz"))
            f1 = float(_p(p, "f1_hz"))
            f_lo, f_hi = min(f0, f1), max(f0, f1)

        elif kind == "bpsk_code":
            p = s["params"]
            # Accept both "chip_rate" and legacy "chip_rate_hz"
            fc = float(_p(p, "carrier_hz", "fc_hz"))
            chip_rate = float(_p(p, "chip_rate", "chip_rate_hz"))
            bw = chip_rate  # approximation: BPSK effective BW ≈ chip rate
            f_lo, f_hi = fc - bw / 2.0, fc + bw / 2.0

        elif kind == "impulses":
            # Wideband in frequency (train-only mode upstream)
            f_lo, f_hi = 0.0, F_total

        else:
            continue

        x_c, y_c, w, h = to_rel_box(t0, t1, f_lo, f_hi)
        lines.append(f"{cls_id(kind)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    return lines
