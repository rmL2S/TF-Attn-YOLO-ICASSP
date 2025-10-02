# Dataset Generation Guide
This document describes how to generate synthetic RF scenarios and convert them into multi-STFT spectrogram datasets with YOLO-style labels.
---
## Contents
- Overview
- Signal primitives (signals.py)
- Scenario synthesis (generate_scenario_dataset)
- Spectrogram dataset (generate_dataset)
- Labels (Ultralytics format)
- Train/Val/Test split
- I/O layout
- Reproducibility
- Quickstart
- STFT configuration examples
- Notes
---
## <a id="overview"></a>Overview
We synthesize time-domain signals (impulses, BPSK, chirps) plus an optional band-limited interference (continuous in time), then export:
1) Scenarios (JSONL): timing + parameters of each emitter/interference.
2) Spectrogram dataset (PT + TXT): multi-STFT magnitude (or log-magnitude) tensors with YOLO boxes.
Key goals: production-ready, torch-native, deterministic, scalable.
---
## <a id="signal-primitives"></a>Signal primitives (signals.py)
### Provided generators
- white_noise / colored_noise: Gaussian noise; colored noise via spectral shaping (1/f^α).
- chirp: real linear chirp sweeping f0→f1 over n samples.
- bpsk_code: rectangular-chip BPSK; optional passband carrier.
- impulses: unmodulated rectangular pulse trains (optional half-cosine edge taper).
- band_noise: band-limited interference via rFFT masking; optional edge taper (bins).
### Utilities
- normalize_rms: set target RMS.
- set_snr_from_noise: scale the signal to reach a target SNR vs a provided noise reference (used for both emitters SNR and interference INR).
Contract: All generators are @torch.no_grad(), support dtype/device, return (tensor, meta) with JSON-serializable metadata.
---
## <a id="scenario-synthesis"></a>Scenario synthesis (generate_scenario_dataset)
### Purpose
Create N independent scenarios. Each scenario is a JSON object containing:
- fs, duration_s, n_samples
- signals: list of emitters with fields:
- kind ∈ {"impulses","bpsk_code","chirp"}
- snr_db, start_sample, duration_samples
- params (type-specific):
- impulses: train-only with prf_hz, duty_cycle, width_s (derived from duty/prf), edge_taper_samples
- bpsk_code: chip_rate (≈ BW), carrier_hz, phase0
- chirp: f0_hz, f1_hz, phase0_rad
- interference (optional): one continuous-in-time band_noise with bands_hz=[(f_lo,f_hi)], inr_db, edge_taper_bins, full_duration=True
- layout: {allow_overlap, guard_hz}
- random_seed
### Constraints (default)
- ≤ 1 interference band anywhere on the spectrum (present for the full acquisition).
- ≤ 3 impulses (very short events).
- ≤ 5 total among chirp + BPSK (medium; chirps medium→large duration/BW).
### Parameter highlights
- Durations (per type) define the local time windows used to place each emitter.
- Bandwidth placement respects guard_hz from DC and Nyquist.
- Non-overlap mode uses greedy packing when allow_overlap=False.
### Output format
- Scenarios are typically exported as JSONL (one JSON per line) for streaming-friendly loading.
---
## <a id="spectrogram-dataset"></a>Spectrogram dataset (generate_dataset)
### Purpose
Convert scenarios (JSONL) into spectrogram samples with labels:
- For each scenario:
1) Create a global background noise (present at all times).
2) For each emitter, synthesize its waveform and scale the signal with set_snr_from_noise to meet its target SNR vs the background noise (local window).
3) If interference is present, synthesize band-limited noise over the full duration and scale it with set_snr_from_noise to meet its INR (i.e., SNR vs the same background).
4) Mix: mixture = background + sum(emitters) + interference.
5) Compute multi-STFT spectrograms using a list of STFT configs (e.g., different nperseg/nfft/noverlap).
6) Save a list of spectrogram tensors (.pt) and a YOLO label file (.txt).
- Split: 80% train, 10% val, 10% test (shuffled with shuffle_seed).
- Progress reporting with tqdm per split.
### STFT
Uses stft_torch(signal, fs, nperseg, noverlap, nfft=None, window=None, one_side=True, scaling=None, apply_padding=True):
- Efficient framing via unfold.
- Real inputs default to one-sided spectrum.
- Optional PSD-like amplitude normalization (scaling="psd").
- Zero-padding per frame to nfft (centered).
- Returns (f, t, S) with S of shape (F, N_frames) or (B, F, N_frames).
### Saved data
- Data: single .pt file per scenario containing List[Tensor] (one tensor per STFT config), each of shape (F, T_frames) and value |S| or log1p(|S|).
- Labels: one .txt file per scenario in Ultralytics format (see below).
---
## <a id="labels"></a>Labels (Ultralytics format)
Classes
0 = impulses, 1 = bpsk_code, 2 = chirp
Box parameterization (normalized) over the time–frequency plane:
- x_center = ((t0 + t1)/2) / duration_s
- y_center = ((f_lo + f_hi)/2) / (fs/2)
- width = (t1 - t0) / duration_s
- height = (f_hi - f_lo) / (fs/2)
Emitter to band mapping
- impulses: broadband → [0, fs/2]
- bpsk_code: [carrier_hz − chip_rate/2, carrier_hz + chip_rate/2]
- chirp: [min(f0_hz,f1_hz), max(f0_hz,f1_hz)]
Each label line:
class x_center y_center width height
(By default, the interference is not labeled; you may add a dedicated class and boxes per band over the full duration if desired.)
---
## <a id="split"></a>Train/Val/Test split
- Shuffle scenarios with shuffle_seed (torch RNG).
- 80/10/10 ratio by index.
- Files are written under train/, val/, test/ subfolders.
---
## <a id="io-layout"></a>I/O layout
```
<out_dir>/
train/
data/ 000123.pt
labels/ 000123.txt
val/
data/ ...
labels/ ...
test/
data/ ...
labels/ ...
```
- *.pt stores a Python list of PyTorch tensors (multi-STFT).
- *.txt stores YOLO labels (one line per emitter).
---
## <a id="reproducibility"></a>Reproducibility
- All sampling uses a local torch.Generator seeded via function arguments.
- Scenario JSON stores seed info to aid repeatability.
- STFT and synthesis run under @torch.no_grad() to avoid accidental grads.
---
## <a id="quickstart"></a>Quickstart
### 1) Generate scenarios (JSONL)
```python
from dataset_scenarios import generate_scenario_dataset
scenarios = generate_scenario_dataset(
n_scenarios=1000,
fs=20_000,
acquisition_duration_s=1.0,
max_impulses=3,
max_chirp_or_bpsk=5,
save_path="data/scenarios_1k.jsonl",
seed=123,
)
```
### 2) Build multi-STFT dataset
```python
from generate_dataset import generate_dataset
import torch
stft_cfgs = [
dict(nperseg=256, nfft=256, noverlap=128, one_side=True, scaling="psd", apply_padding=True),
dict(nperseg=512, nfft=512, noverlap=256, one_side=True, scaling="psd", apply_padding=True),
]
generate_dataset(
scenarios_path="data/scenarios_1k.jsonl",
out_dir="data/spectro",
stft_configs=stft_cfgs,
device="cpu",
dtype=torch.float32,
shuffle_seed=123,
log_magnitude=True,
)
```
---
## <a id="stft-configs"></a>STFT configuration examples
### Balanced time–freq resolution
```python
dict(nperseg=512, nfft=512, noverlap=384, one_side=True, scaling="psd", apply_padding=True)
```
### Higher time resolution
```python
dict(nperseg=256, nfft=256, noverlap=192, one_side=True, scaling="psd", apply_padding=True)
```
### Wider bandwidth capture (zero-pad to nfft)
```python
dict(nperseg=384, nfft=1024, noverlap=256, one_side=True, scaling="psd", apply_padding=True)
```
---
## Notes
- Magnitudes are saved as log1p(|S|) by default. Set log_magnitude=False to store |S|.
- Interference power is controlled via INR = SNR vs the background noise (continuous in time, band-limited in frequency).
- Interference is not labeled by default (only emitters are); optionally add a dedicated class if needed.
- Label frequency normalization assumes one-sided spectra ([0, fs/2]). If you switch to two-sided STFT, adapt label normalization accordingly.