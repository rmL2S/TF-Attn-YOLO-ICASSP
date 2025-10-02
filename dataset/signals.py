
from typing import Dict, Literal, Optional, Tuple, Iterable, Union
import torch

Tensor = torch.Tensor
NoiseColor = Literal["white", "pink", "brown", "custom"]
SignalKind = Literal["white_noise", "colored_noise", "chirp", "bpsk_code", "tone", "impulses"]

__all__ = [
    "white_noise",
    "colored_noise",
    "chirp",
    "bpsk_code",
    "tone",
    "impulses",
    "generate",
    "normalize_rms",
    "set_snr_from_noise",  
]

# ---------------------------- Utilities ----------------------------

def _gen(seed: Optional[int]) -> Optional[torch.Generator]:
    """CPU generator for reproducible randomness."""
    if seed is None:
        return None
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g

def _to(x: Tensor, dtype: torch.dtype, device: Union[torch.device, str]) -> Tensor:
    """Minimal, safe cast to dtype/device."""
    d = torch.device(device)
    if x.device != d:
        x = x.to(d)
    if x.dtype != dtype:
        x = x.to(dtype)
    return x

@torch.no_grad()
def normalize_rms(x: Tensor, target_rms: float = 1.0, eps: float = 1e-12) -> Tensor:
    """Scale x to a target RMS."""
    if x.numel() == 0:
        raise ValueError("normalize_rms: empty tensor.")
    rms = torch.sqrt(torch.mean(x.float().pow(2))) + eps
    return x * (float(target_rms) / rms)

@torch.no_grad()
def set_snr_from_noise(signal: torch.Tensor, noise_ref: torch.Tensor, snr_db: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Scale 'signal' so that Ps/Pn == 10^(snr_db/10), where:
      - Ps is power of the (scaled) signal
      - Pn is power of the provided noise_ref (background noise on same window)
    """
    Pn = torch.mean(noise_ref.float().pow(2)) + eps
    target_Ps = Pn * (10.0 ** (float(snr_db) / 10.0))
    Ps = torch.mean(signal.float().pow(2)) + eps
    scale = torch.sqrt(target_Ps / Ps)
    return signal * scale

# ---------------------------- Generators (one sample) ----------------------------

@torch.no_grad()
def white_noise(
    n: int,
    *,
    std: float = 1.0,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[Tensor, Dict]:
    """Gaussian white noise ~ N(0,std^2), length n."""
    if n <= 0:
        raise ValueError("white_noise: n must be > 0.")
    x = torch.randn(n, dtype=torch.float32, generator=_gen(seed)) * float(std)
    x = _to(x, dtype, device)
    return x, {"type": "white_noise", "std": std, "seed": seed}

@torch.no_grad()
def colored_noise(
    n: int,
    *,
    color: NoiseColor = "pink",
    alpha: Optional[float] = None,
    std: float = 1.0,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[Tensor, Dict]:
    """
    Colored noise via spectral shaping on rFFT half-spectrum:
      |H(f)| ∝ 1 / f^(alpha/2).
    color: 'pink'(1), 'brown'(2), 'white'(0), or 'custom' with alpha>=0.
    """
    if n <= 0:
        raise ValueError("colored_noise: n must be > 0.")
    if color == "white":
        return white_noise(n, std=std, seed=seed, dtype=dtype, device=device)

    if color == "pink":
        alpha_eff = 1.0
    elif color == "brown":
        alpha_eff = 2.0
    elif color == "custom":
        if alpha is None or alpha < 0:
            raise ValueError("colored_noise: provide alpha>=0 for color='custom'.")
        alpha_eff = float(alpha)
    else:
        raise ValueError(f"colored_noise: unknown color '{color}'.")

    g = _gen(seed)
    # Random complex half-spectrum (rFFT bins), complex64
    real = torch.randn(n // 2 + 1, dtype=torch.float32, generator=g)
    imag = torch.randn(n // 2 + 1, dtype=torch.float32, generator=g)
    Xh = torch.complex(real, imag)

    # Magnitude shaping (skip DC to avoid div-by-zero)
    freqs = torch.fft.rfftfreq(n, d=1.0)  # normalized bin spacing
    mag = torch.ones_like(freqs)
    if n >= 2:
        mag[1:] = 1.0 / torch.pow(freqs[1:], alpha_eff / 2.0)
    Xh = Xh * mag.to(Xh.dtype)

    # Real signal
    x = torch.fft.irfft(Xh, n=n)
    x = normalize_rms(x, target_rms=std)
    x = _to(x, dtype, device)
    return x, {"type": "colored_noise", "color": color, "alpha": alpha_eff, "std": std, "seed": seed}

@torch.no_grad()
def chirp(
    n: int,
    *,
    fs: float,
    f0: float,
    f1: float,
    phase0: float = 0.0,
    amplitude: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[Tensor, Dict]:
    """Real cosine linear chirp sweeping f0→f1 over n samples at sampling fs."""
    if n <= 0:
        raise ValueError("chirp: n must be > 0.")
    if fs <= 0:
        raise ValueError("chirp: fs must be > 0.")
    d = torch.device(device)
    t = torch.arange(n, dtype=torch.float64, device=d) / float(fs)
    k = (float(f1) - float(f0)) / (n / float(fs))  # Hz/s
    phi = 2.0 * torch.pi * (float(f0) * t + 0.5 * k * t * t) + float(phase0)
    x = float(amplitude) * torch.cos(phi)
    x = _to(x, dtype, d)
    return x, {
        "type": "chirp_linear",
        "fs": fs,
        "f0": f0,
        "f1": f1,
        "phase0": phase0,
        "amplitude": amplitude,
    }

@torch.no_grad()
def bpsk_code(
    n: int,
    *,
    fs: float,
    chip_rate: float,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
    carrier_hz: Optional[float] = None,
    phase0: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[Tensor, Dict]:
    """
    Random BPSK (±1) with rectangular chip shaping.
    If carrier_hz is set, modulates a cosine carrier. Output length is exactly n.
    """
    if n <= 0:
        raise ValueError("bpsk_code: n must be > 0.")
    if fs <= 0 or chip_rate <= 0:
        raise ValueError("bpsk_code: fs and chip_rate must be > 0.")
    d = torch.device(device)
    g = _gen(seed)

    chip_len = int(torch.floor(torch.tensor(fs / chip_rate)).item())
    if chip_len < 1:
        raise ValueError("bpsk_code: chip_rate too high for given fs (chip_len < 1).")

    chips = int(torch.ceil(torch.tensor(n / chip_len)).item())
    symbols = torch.randint(0, 2, (chips,), generator=g, dtype=torch.int64)
    symbols = symbols.to(torch.float32).mul_(2.0).add_(-1.0)  # {0,1} -> {-1,+1}
    x = torch.repeat_interleave(symbols, repeats=chip_len)[:n] * float(amplitude)
    x = x.to(d)

    if carrier_hz is not None and carrier_hz != 0.0:
        t = torch.arange(n, dtype=torch.float64, device=d) / float(fs)
        x = x * torch.cos(2.0 * torch.pi * float(carrier_hz) * t + float(phase0))

    x = _to(x, dtype, d)
    return x, {
        "type": "bpsk_code",
        "fs": fs,
        "chip_rate": chip_rate,
        "amplitude": amplitude,
        "seed": seed,
        "carrier_hz": carrier_hz,
        "phase0": phase0,
    }

# ------------------------------ Band-limited noise ------------------------------
@torch.no_grad()
def band_noise(
    n: int,
    *,
    fs: float,
    bands_hz: Union[Tuple[float, float], Iterable[Tuple[float, float]]],
    std: float = 1.0,
    edge_taper_bins: int = 0,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[Tensor, Dict]:
    """
    Band-limited white noise (real-valued) synthesized by shaping the rFFT half-spectrum.

    Args:
      n: number of samples (time domain).
      fs: sampling frequency (Hz).
      bands_hz: one (f_lo, f_hi) or a list of bands (Hz), each within (0, fs/2).
      std: target RMS in time domain after synthesis.
      edge_taper_bins: optional cosine taper width (bins) at each band edge to soften transitions.
      seed: RNG seed for reproducibility.
      dtype/device: output specs.

    Returns:
      x: Tensor (n,) real, time-domain band-limited noise.
      meta: JSON-serializable dict of parameters.
    """
    if n <= 0:
        raise ValueError("band_noise: n must be > 0.")
    if fs <= 0:
        raise ValueError("band_noise: fs must be > 0.")

    d = torch.device(device)
    g = _gen(seed)

    # Normalize input bands to a list
    if isinstance(bands_hz, tuple):
        bands = [bands_hz]
    else:
        bands = list(bands_hz)

    if len(bands) == 0:
        raise ValueError("band_noise: bands_hz must contain at least one band.")

    nyq = fs / 2.0
    # rFFT bins: 0..N/2 inclusive (F = n//2 + 1)
    F = n // 2 + 1
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs)  # Hz, shape (F,)
    mask = torch.zeros(F, dtype=torch.float32)

    # Build mask with optional edge tapers
    for (flo, fhi) in bands:
        if not (0.0 <= flo < fhi <= nyq):
            raise ValueError(f"band_noise: invalid band ({flo}, {fhi}) for fs={fs}.")
        # hard passband region indices
        inb = (freqs >= flo) & (freqs <= fhi)
        mask = torch.maximum(mask, inb.to(mask.dtype))

        # optional taper on both sides to reduce ringing (cosine ramp)
        if edge_taper_bins > 0:
            # left edge
            left_idx = torch.nonzero(freqs >= flo, as_tuple=False)
            right_idx = torch.nonzero(freqs <= fhi, as_tuple=False)
            if left_idx.numel() > 0:
                li = int(left_idx[0].item())
                l0 = max(0, li - edge_taper_bins)
                if l0 < li:
                    t = torch.linspace(0, 1, steps=li - l0)
                    mask[l0:li] = torch.maximum(mask[l0:li], 0.5 - 0.5 * torch.cos(torch.pi * t))
            if right_idx.numel() > 0:
                ri = int(right_idx[-1].item())
                r1 = min(F - 1, ri + edge_taper_bins)
                if ri < r1:
                    t = torch.linspace(0, 1, steps=r1 - ri + 1)
                    # descending ramp
                    ramp = 0.5 + 0.5 * torch.cos(torch.pi * t)
                    mask[ri:r1 + 1] = torch.maximum(mask[ri:r1 + 1], ramp)

    # Random complex half-spectrum (white) then shape by sqrt(mask) so PSD ~ mask
    real = torch.randn(F, dtype=torch.float32, generator=g)
    imag = torch.randn(F, dtype=torch.float32, generator=g)
    Xh = torch.complex(real, imag)

    # Apply magnitude mask (amplitude shaping)
    Xh = Xh * mask.to(Xh.dtype).sqrt()

    # Back to time-domain real noise in the specified bands
    x = torch.fft.irfft(Xh, n=n)  # (n,)
    x = normalize_rms(x, target_rms=std)
    x = _to(x, dtype, d)

    meta = {
        "type": "band_noise",
        "fs": fs,
        "bands_hz": [(float(a), float(b)) for (a, b) in bands],
        "std": std,
        "edge_taper_bins": int(edge_taper_bins),
        "seed": seed,
    }
    return x, meta

# ---------------------------- Convenient mixer ----------------------------

@torch.no_grad()
def add_interference(
    x: Tensor,
    *,
    fs: float,
    bands_hz: Union[Tuple[float, float], Iterable[Tuple[float, float]]],
    sir_db: Optional[float] = None,
    std: Optional[float] = None,
    edge_taper_bins: int = 0,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """
    Synthesize band-limited noise on given band(s) and add it to 'x'.

    Args:
      x: reference time signal (n,).
      fs: sampling frequency.
      bands_hz: one or multiple bands in Hz.
      sir_db: if provided, scales interference to reach this SIR vs 'x'.
      std: if provided, sets interference RMS before optional SIR scaling.
      edge_taper_bins: cosine taper at band edges in rFFT bins.
      seed: RNG seed.

    Returns:
      x_mix: x + i_scaled
      i_scaled: the interference alone (scaled)
      meta: dict with synthesis details
    """
    if x.ndim != 1:
        raise ValueError("add_interference: x must be 1-D.")
    n = x.shape[0]

    # 1) Generate interference with unit RMS (safer), then optional std
    i, meta_i = band_noise(
        n,
        fs=fs,
        bands_hz=bands_hz,
        std=1.0,
        edge_taper_bins=edge_taper_bins,
        seed=seed,
        dtype=x.dtype,
        device=x.device,
    )
    if std is not None:
        i = normalize_rms(i, target_rms=float(std))

    # 2) Optional SIR scaling relative to x
    if sir_db is not None:
        _, i = set_sir(x, i, sir_db=float(sir_db))

    x_mix = x + i
    meta = {
        "interference": meta_i,
        "sir_db": float(sir_db) if sir_db is not None else None,
        "interf_rms": float(torch.sqrt(torch.mean(i.float().pow(2))).item()),
    }
    return x_mix, i, meta


@torch.no_grad()
def impulses(
    n: int,
    *,
    fs: float,
    # Définition de la largeur
    width_s: Optional[float] = None,          # largeur d'une impulsion en secondes
    width_samples: Optional[int] = None,      # ou en échantillons
    amplitude: float = 1.0,
    # Placement: soit positions explicites, soit train d'impulsions via PRF
    starts_s: Optional[Iterable[float]] = None,   # positions en s (prioritaire si fourni)
    starts_samples: Optional[Iterable[int]] = None,
    prf_hz: Optional[float] = None,               # sinon: fréquence de répétition
    duty_cycle: Optional[float] = None,           # 0..1 ; si fourni -> width = duty_cycle / prf
    num_pulses: Optional[int] = None,             # nb impulsons pour le train
    # Finition
    edge_taper_samples: int = 0,                  # adoucit chaque bord (cosine) en nb d'échantillons
    dtype: torch.dtype = torch.float32,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[Tensor, Dict]:
    """
    Generate unmodulated rectangular impulses (baseband), as a single pulse or a pulse train.

    Modes:
      1) Explicit positions: provide starts_{s|samples} and a width (width_s or width_samples).
      2) Pulse train: provide prf_hz; width from width_{s|samples} or from duty_cycle/prf.
         num_pulses optional (defaults to max that fits in n).

    edge_taper_samples > 0 applies a half-cosine ramp on each edge to reduce spectral ringing.
    """
    if n <= 0:
        raise ValueError("impulses: n must be > 0.")
    if fs <= 0:
        raise ValueError("impulses: fs must be > 0.")
    d = torch.device(device)

    # --- largeur de l'impulsion (en échantillons) ---
    if width_samples is not None and width_samples <= 0:
        raise ValueError("impulses: width_samples must be > 0 when provided.")
    if width_samples is None:
        if width_s is None and duty_cycle is None:
            raise ValueError("impulses: provide width_s/width_samples or duty_cycle with prf_hz.")
        if width_s is not None:
            width_samples = max(1, int(round(float(width_s) * fs)))
    # si duty_cycle fourni + prf_hz -> définir/écraser la largeur
    if duty_cycle is not None:
        if prf_hz is None or prf_hz <= 0:
            raise ValueError("impulses: duty_cycle requires prf_hz > 0.")
        if not (0.0 < duty_cycle < 1.0):
            raise ValueError("impulses: duty_cycle must be in (0,1).")
        period_s = 1.0 / float(prf_hz)
        width_samples = max(1, int(round(period_s * duty_cycle * fs)))

    if width_samples is None or width_samples <= 0:
        raise ValueError("impulses: could not determine a positive width in samples.")

    # --- positions de départ (échantillons) ---
    starts_list: list[int] = []
    if starts_samples is not None or starts_s is not None:
        if starts_samples is not None:
            starts_list = [int(s) for s in starts_samples]
        else:
            starts_list = [int(round(float(s) * fs)) for s in starts_s]  # type: ignore[arg-type]
    else:
        # Train d'impulsions
        if prf_hz is None or prf_hz <= 0:
            # Par défaut: une seule impulsion centrée
            starts_list = [max(0, (n - width_samples) // 2)]
        else:
            period_samples = max(1, int(round(fs / float(prf_hz))))
            if num_pulses is None:
                # fit-as-many-as-possible
                num_pulses = 1 + max(0, (n - width_samples) // period_samples)
            # positions régulières à partir de 0
            starts_list = [k * period_samples for k in range(int(num_pulses))]

    # clamp et filtrage des positions valides
    valid_starts: list[int] = []
    for s in starts_list:
        if s >= n or s + width_samples <= 0:
            continue
        valid_starts.append(max(0, min(int(s), n - 1)))
    if not valid_starts:
        # rien ne rentre : renvoyer zéro + meta
        x = torch.zeros(n, dtype=dtype, device=d)
        return x, {
            "type": "impulses",
            "fs": fs,
            "starts_samples": [],
            "width_samples": int(width_samples),
            "amplitude": float(amplitude),
            "edge_taper_samples": int(edge_taper_samples),
            "note": "no valid pulse fit within n samples",
        }

    # --- construction du signal ---
    x = torch.zeros(n, dtype=torch.float32, device=d)
    w = int(width_samples)

    # fenêtre rectangulaire de base + taper optionnel
    if edge_taper_samples > 0:
        et = int(edge_taper_samples)
        et = max(0, min(et, w // 2))
        core = w - 2 * et
        if core < 0:
            et = w // 2
            core = w - 2 * et
        # ramp up (half-cosine)
        if et > 0:
            up = 0.5 * (1.0 - torch.cos(torch.linspace(0, torch.pi, steps=et, device=d)))
        else:
            up = torch.empty(0, device=d)
        # flat core
        flat = torch.ones(core, device=d)
        # ramp down
        if et > 0:
            down = torch.flip(up, dims=[0])
        else:
            down = torch.empty(0, device=d)
        kernel = torch.cat([up, flat, down])  # shape (w,)
    else:
        kernel = torch.ones(w, device=d)

    kernel = kernel * float(amplitude)

    # ajouter chaque impulsion
    for s in valid_starts:
        e = min(n, s + w)
        ks = 0
        ke = e - s
        # tronquer kernel si dépasse la fin
        x[s:e] += kernel[ks:ke]

    x = _to(x, dtype, d)

    meta = {
        "type": "impulses",
        "fs": fs,
        "starts_samples": [int(s) for s in valid_starts],
        "width_samples": int(w),
        "amplitude": float(amplitude),
        "edge_taper_samples": int(edge_taper_samples),
        "mode": "explicit" if (starts_samples is not None or starts_s is not None) else ("train" if prf_hz else "single"),
        "prf_hz": float(prf_hz) if prf_hz else None,
        "duty_cycle": float(duty_cycle) if duty_cycle is not None else None,
        "num_pulses": int(num_pulses) if num_pulses is not None else None,
    }
    return x, meta

