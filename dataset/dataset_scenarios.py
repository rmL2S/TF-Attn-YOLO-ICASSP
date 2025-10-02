# dataset_scenarios.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Literal
import json
import math
import os
import torch

SignalKind = Literal["chirp", "bpsk_code", "impulses"]

__all__ = ["generate_scenario_dataset"]


@torch.no_grad()
def generate_scenario_dataset(
    n_scenarios: int,
    *,
    fs: float,
    acquisition_duration_s: float,
    # Caps
    max_impulses: int = 3,
    max_chirp_or_bpsk: int = 5,
    # Global SNR
    snr_db_range: Tuple[float, float] = (-10.0, 20.0),
    # Impulses (very short)
    impulses_width_s_range: Tuple[float, float] = (0.2e-3, 2e-3),
    impulses_train_probability: float = 0.3,
    impulses_prf_hz_range: Tuple[float, float] = (50.0, 300.0),
    impulses_duty_cycle_range: Tuple[float, float] = (0.03, 0.15),
    impulses_edge_taper_samples_range: Tuple[int, int] = (0, 8),
    impulses_event_duration_s_range: Tuple[float, float] = (2e-3, 20e-3),
    # BPSK (medium)
    bpsk_event_duration_s_range: Tuple[float, float] = (20e-3, 120e-3),
    bpsk_bw_hz_range: Tuple[float, float] = (400.0, 2_000.0),
    # Chirp (medium→large)
    chirp_event_duration_s_range: Tuple[float, float] = (40e-3, 250e-3),
    chirp_bw_hz_range: Tuple[float, float] = (800.0, 4_500.0),
    # Interference (≤1 band anywhere)
    include_interference_probability: float = 0.5,
    interference_inr_db_range: Tuple[float, float] = (0.0, 15.0),
    interference_bandwidth_hz_range: Tuple[float, float] = (300.0, 8_000.0),
    interference_edge_taper_bins: int = 2,
    # Placement
    guard_hz: float = 50.0,
    allow_overlap: bool = True,
    # RNG & export
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
) -> List[Dict]:
    """
    Build a dataset of scenario specifications (JSON-serializable dicts).

    Each scenario contains up to:
      - 1 band-limited noise interference (optional, anywhere in spectrum).
      - 3 unmodulated impulse emitters (very short events).
      - 5 emitters in total among chirp and BPSK (medium; chirp medium→large in duration/BW).

    Args:
        n_scenarios: Number of scenarios to generate.
        fs: Sampling rate (Hz).
        acquisition_duration_s: Total acquisition duration (s).
        max_impulses: Max number of impulse emitters.
        max_chirp_or_bpsk: Max combined number of chirp and BPSK emitters.
        snr_db_range: Per-signal SNR range [min, max] (dB).
        impulses_width_s_range: Single-pulse width range (s).
        impulses_train_probability: Probability to generate a pulse train instead of a single pulse.
        impulses_prf_hz_range: Pulse-train PRF range (Hz).
        impulses_duty_cycle_range: Pulse-train duty-cycle range (0,1).
        impulses_edge_taper_samples_range: Half-cosine edge taper range (samples).
        impulses_event_duration_s_range: Time-window range reserved per impulse event (s).
        bpsk_event_duration_s_range: Time-window range per BPSK event (s).
        bpsk_bw_hz_range: BPSK effective bandwidth (≈chip rate) range (Hz).
        chirp_event_duration_s_range: Time-window range per chirp event (s).
        chirp_bw_hz_range: Chirp bandwidth range (Hz).
        include_interference_probability: Probability to include one interference band.
        interference_sir_db_range: SIR range [min, max] (dB) vs the sum of all signals.
        interference_bandwidth_hz_range: Interference bandwidth range (Hz).
        interference_edge_taper_bins: Cosine taper width (FFT bins) at band edges.
        guard_hz: Guard band kept away from DC and Nyquist (Hz).
        allow_overlap: Allow time overlaps between signals.
        seed: RNG seed.
        save_path: Optional JSONL path to write the dataset.

    Returns:
        List of scenario dicts.
    """
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be > 0.")
    if fs <= 0 or acquisition_duration_s <= 0:
        raise ValueError("fs and acquisition_duration_s must be > 0.")
    if guard_hz < 0:
        raise ValueError("guard_hz must be >= 0.")
    if not (0.0 <= include_interference_probability <= 1.0):
        raise ValueError("include_interference_probability must be in [0,1].")
    if snr_db_range[0] > snr_db_range[1]:
        raise ValueError("snr_db_range is invalid.")
    if impulses_width_s_range[0] <= 0 or impulses_width_s_range[0] > impulses_width_s_range[1]:
        raise ValueError("impulses_width_s_range is invalid.")
    if impulses_prf_hz_range[0] <= 0 or impulses_prf_hz_range[0] > impulses_prf_hz_range[1]:
        raise ValueError("impulses_prf_hz_range is invalid.")
    if not (0.0 < impulses_duty_cycle_range[0] < impulses_duty_cycle_range[1] < 1.0):
        raise ValueError("impulses_duty_cycle_range must be within (0,1).")
    if impulses_edge_taper_samples_range[0] < 0 or impulses_edge_taper_samples_range[0] > impulses_edge_taper_samples_range[1]:
        raise ValueError("impulses_edge_taper_samples_range is invalid.")
    if bpsk_bw_hz_range[0] <= 0 or bpsk_bw_hz_range[0] > bpsk_bw_hz_range[1]:
        raise ValueError("bpsk_bw_hz_range is invalid.")
    if chirp_bw_hz_range[0] <= 0 or chirp_bw_hz_range[0] > chirp_bw_hz_range[1]:
        raise ValueError("chirp_bw_hz_range is invalid.")
    if interference_edge_taper_bins < 0:
        raise ValueError("interference_edge_taper_bins must be >= 0.")

    nyq = fs / 2.0
    if guard_hz * 2.0 >= nyq:
        raise ValueError("guard_hz too large for fs.")
    n_total = int(round(acquisition_duration_s * fs))
    if n_total <= 0:
        raise ValueError("acquisition_duration_s * fs yields zero samples.")

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))

    def runif(a: float, b: float) -> float:
        return float((torch.rand(1, generator=gen) * (b - a) + a).item())

    def rint(low: int, high_inclusive: int) -> int:
        if high_inclusive < low:
            return low
        return int(torch.randint(low, high_inclusive + 1, (1,), generator=gen).item())

    def rbernoulli(p: float) -> bool:
        return bool(torch.rand(1, generator=gen).item() < p)

    placed: List[Tuple[int, int]] = []

    def overlaps(s: int, d: int) -> bool:
        e = s + d
        for (ps, pd) in placed:
            pe = ps + pd
            if not (e <= ps or s >= pe):
                return True
        return False

    def reserve(s: int, d: int) -> None:
        placed.append((s, d))

    def pick_time_window(dmin: float, dmax: float) -> Tuple[int, int]:
        dur_s = max(1.0 / fs, runif(dmin, dmax))
        dur_n = int(max(1, round(dur_s * fs)))
        dur_n = min(dur_n, n_total)
        if allow_overlap:
            return rint(0, max(0, n_total - dur_n)), dur_n
        for _ in range(32):
            s = rint(0, max(0, n_total - dur_n))
            if not overlaps(s, dur_n):
                reserve(s, dur_n)
                return s, dur_n
        for s in range(0, n_total - dur_n + 1):
            if not overlaps(s, dur_n):
                reserve(s, dur_n)
                return s, dur_n
        return 0, min(dur_n, n_total)

    def chirp_params() -> Dict:
        bw = runif(*chirp_bw_hz_range)
        up = rint(0, 1) == 1
        f_lo = guard_hz + bw / 2.0
        f_hi = (nyq - guard_hz) - bw / 2.0
        f_hi = max(f_hi, f_lo)
        fc = runif(f_lo, f_hi)
        f0, f1 = fc - bw / 2.0, fc + bw / 2.0
        if not up:
            f0, f1 = f1, f0
        return {"fs": fs, "f0_hz": f0, "f1_hz": f1, "phase0_rad": runif(0.0, 2.0 * math.pi), "amplitude": 1.0}

    def bpsk_params() -> Dict:
        bw = runif(*bpsk_bw_hz_range)
        chip_rate = bw  
        half = bw / 2.0
        f_lo = guard_hz + half
        f_hi = (nyq - guard_hz) - half
        f_hi = max(f_hi, f_lo)
        fc = runif(f_lo, f_hi)
        return {
            "fs": fs,
            "chip_rate": chip_rate,    
            "carrier_hz": fc,
            "phase0": runif(0.0, 2.0 * math.pi), 
            "amplitude": 1.0,
        }

    def impulses_params() -> Dict:
        """Return parameters for a pulse train."""
        taper = rint(*impulses_edge_taper_samples_range)
        prf = runif(*impulses_prf_hz_range)
        duty = runif(*impulses_duty_cycle_range)
        width_s = max(1.0 / fs, duty / prf)

        return {
            "fs": fs,
            "prf_hz": prf,
            "duty_cycle": duty,
            "width_s": width_s,
            "amplitude": 1.0,
            "edge_taper_samples": int(taper),
        }


    def interference_block() -> Optional[Dict]:
        if not rbernoulli(include_interference_probability):
            return None
        inr_db = runif(*interference_inr_db_range)  # <-- INR
        bw = runif(*interference_bandwidth_hz_range)
        bw = min(bw, max(1.0, nyq - 2.0 * guard_hz))
        half = bw / 2.0
        f_lo = guard_hz + half
        f_hi = (nyq - guard_hz) - half
        f_hi = max(f_hi, f_lo)
        fc = runif(f_lo, f_hi)
        band = (fc - half, fc + half)
        return {
            "present": True,
            "kind": "band_noise",
            "inr_db": inr_db,                    # <-- clé renommée
            "bands_hz": [band],
            "edge_taper_bins": int(interference_edge_taper_bins),
            "full_duration": True
        }

    def build_one() -> Dict:
        nonlocal placed
        placed = []
        n_imp = rint(0, max_impulses)
        n_cb = rint(0, max_chirp_or_bpsk)
        n_chirp = rint(0, n_cb)
        n_bpsk = n_cb - n_chirp

        signals: List[Dict] = []
        for _ in range(n_imp):
            start, dur = pick_time_window(*impulses_event_duration_s_range)
            signals.append({
                "kind": "impulses",
                "snr_db": runif(*snr_db_range),
                "start_sample": int(start),
                "duration_samples": int(dur),
                "params": impulses_params()
            })

        for _ in range(n_bpsk):
            start, dur = pick_time_window(*bpsk_event_duration_s_range)
            signals.append({"kind": "bpsk_code", "snr_db": runif(*snr_db_range), "start_sample": int(start), "duration_samples": int(dur), "params": bpsk_params()})
        for _ in range(n_chirp):
            start, dur = pick_time_window(*chirp_event_duration_s_range)
            signals.append({"kind": "chirp", "snr_db": runif(*snr_db_range), "start_sample": int(start), "duration_samples": int(dur), "params": chirp_params()})

        return {
            "fs": fs,
            "duration_s": acquisition_duration_s,
            "n_samples": n_total,
            "signals": signals,
            "interference": interference_block(),
            "layout": {"allow_overlap": allow_overlap, "guard_hz": guard_hz},
            "random_seed": int(seed) if seed is not None else None,
        }

    dataset: List[Dict] = [build_one() for _ in range(n_scenarios)]

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for scn in dataset:
                f.write(json.dumps(scn) + "\n")

    return dataset
