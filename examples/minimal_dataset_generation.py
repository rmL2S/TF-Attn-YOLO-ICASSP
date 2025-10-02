from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from dataset import generate_scenario_dataset, generate_dataset


def _parse_stft_list(values: List[str]) -> List[Dict]:
    """Parse 'nperseg,nfft,noverlap' into STFT dicts with sane defaults."""
    cfgs: List[Dict] = []
    for v in values:
        parts = [p.strip() for p in v.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid STFT spec '{v}'. Expected 'nperseg,nfft,noverlap'.")
        nperseg, nfft, noverlap = map(int, parts)
        if not (nperseg > 0 and nfft > 0 and 0 <= noverlap < nperseg):
            raise ValueError(f"Invalid STFT triple: {v}")
        cfgs.append(dict(
            nperseg=nperseg,
            nfft=nfft,
            noverlap=noverlap,
            one_side=True,
            scaling="psd",
            apply_padding=True,
        ))
    return cfgs


def _default_stft_configs() -> List[Dict]:
    return [
        dict(nperseg=256, nfft=256, noverlap=0, one_side=True, scaling="psd", apply_padding=True),
        dict(nperseg=512, nfft=512, noverlap=0, one_side=True, scaling="psd", apply_padding=True),
    ]


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("float32", "fp32"):
        return torch.float32
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype '{s}'")


def run(
    *,
    out_root: Path,
    n_scenarios: int,
    fs: float,
    acquisition_duration_s: float,
    max_impulses: int,
    max_chirp_or_bpsk: int,
    stft_configs: Iterable[Dict],
    device: str,
    dtype: torch.dtype,
    seed: Optional[int],
    include_interference_probability: float,
) -> None:
    """
    1) Write scenarios JSONL -> <out_root>/scenarios.jsonl
    2) Build spectrogram dataset -> <out_root>/{train,val,test}/{data,labels}
    """
    out_root.mkdir(parents=True, exist_ok=True)
    scenarios_path = out_root / "scenarios.jsonl"

    # --- Scenarios ---
    generate_scenario_dataset(
        n_scenarios=n_scenarios,
        fs=fs,
        acquisition_duration_s=acquisition_duration_s,
        max_impulses=max_impulses,
        max_chirp_or_bpsk=max_chirp_or_bpsk,
        include_interference_probability=include_interference_probability,
        save_path=str(scenarios_path),
        seed=seed,
    )

    # --- Spectrogram dataset ---
    generate_dataset(
        scenarios_path=str(scenarios_path),
        out_dir=str(out_root),
        stft_configs=stft_configs,
        device=device,
        dtype=dtype,
        shuffle_seed=seed if seed is not None else 123,
        log_magnitude=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate scenarios then multi-STFT dataset + YOLO labels."
    )
    p.add_argument("--out", type=str, required=True, help="Output root directory.")
    p.add_argument("--n_scenarios", type=int, default=1000)
    p.add_argument("--fs", type=float, default=20_000.0)
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--max_impulses", type=int, default=3)
    p.add_argument("--max_chirp_or_bpsk", type=int, default=5)
    p.add_argument("--stft", type=str, action="append",
                   help="STFT config 'nperseg,nfft,noverlap'. May be used multiple times.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--interference_p", type=float, default=0.5)

    args = p.parse_args()
    stft_cfgs = _parse_stft_list(args.stft) if args.stft else _default_stft_configs()

    print('stft_cfgs ->', stft_cfgs)
    print('duration ->', args.duration)
    print('fs ->', args.fs)
    print('n points ->', args.fs*args.duration)

    run(
        out_root=Path(args.out),
        n_scenarios=int(args.n_scenarios),
        fs=float(args.fs),
        acquisition_duration_s=float(args.duration),
        max_impulses=int(args.max_impulses),
        max_chirp_or_bpsk=int(args.max_chirp_or_bpsk),
        stft_configs=stft_cfgs,
        device=args.device,
        dtype=_dtype_from_str(args.dtype),
        seed=int(args.seed) if args.seed is not None else None,
        include_interference_probability=float(args.interference_p),
    )


if __name__ == "__main__":
    main()
