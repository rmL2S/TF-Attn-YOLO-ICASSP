# dataset/stft.py
import torch
import torch.nn.functional as F
from typing import Union, Tuple

@torch.no_grad()
def stft_torch(
    signal: torch.Tensor,
    fs: float,
    *,
    nperseg: int = 256,
    noverlap: int = 0,
    nfft: int = None,
    window: str = "hann",
    one_side: bool = True,     # compat avec ton cfg
    scaling: str = "psd",      # conservé si tu normalises ensuite
    apply_padding: bool = True,
    pad_mode: str = "reflect",
    dtype: torch.dtype = torch.float32,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Production STFT using torch.stft.
    - win_length == nperseg
    - hop_length == nperseg - noverlap
    - Pad auto si len(signal) < nperseg  (quand apply_padding=True)
    """
    x = signal.to(device=device, dtype=dtype).flatten()  # [T]

    if nperseg <= 0:
        raise ValueError("nperseg must be > 0.")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("0 <= noverlap < nperseg must hold.")
    hop = int(nperseg - noverlap)

    # Fenêtre
    if window == "hann":
        win = torch.hann_window(nperseg, periodic=True, dtype=dtype, device=device)
    elif window == "hamming":
        win = torch.hamming_window(nperseg, periodic=True, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported window: {window!r}")

    # nfft par défaut: au moins nperseg (puissance de 2 si tu veux)
    if nfft is None:
        nfft = max(nperseg, 1 << (nperseg - 1).bit_length())
    if nfft < nperseg:
        raise ValueError("nfft must be >= nperseg.")

    # Si signal trop court, pad à droite pour garantir au moins 1 frame complète
    if x.numel() < nperseg:
        if not apply_padding:
            raise ValueError(
                f"Signal length ({x.numel()}) < nperseg ({nperseg}) and apply_padding=False."
            )
        x = F.pad(x, (0, nperseg - x.numel()))

    # torch.stft fait le framing correct (win_length=nperseg, hop_length=hop)
    X = torch.stft(
        x,
        n_fft=nfft,
        hop_length=hop,
        win_length=nperseg,
        window=win,
        center=True,                # équivalent à padding centré
        pad_mode=pad_mode,
        normalized=False,           # on peut gérer scaling après
        onesided=bool(one_side),
        return_complex=True,
    )  # shape: [freq_bins, frames]

    # Axes f, t
    f_bins = X.shape[0]
    t_frames = X.shape[1]
    f = torch.linspace(0.0, fs / 2.0 if one_side else fs, f_bins, device=device, dtype=torch.float32)
    # centres temporels (avec center=True, frames centrées)
    # temps = (frame_idx * hop) / fs
    t = (torch.arange(t_frames, device=device, dtype=torch.float32) * hop) / float(fs)

    # Magnitude (tu peux ajuster selon 'scaling')
    S = X.abs()

    # Optionnel: normalisation "psd"
    if scaling == "psd":
        # Échelle ~ énergie par bin (simple, rapide)
        # Note: ajuste selon tes besoins (scipy.signal.welch a une normalisation plus détaillée)
        scale = (win.pow(2).sum()).sqrt()
        S = S / (scale + 1e-12)

    return f, t, S
