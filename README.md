# TF-Attn-YOLO-ICASSP

Official implementation of **TF-Attn-YOLO** — a YOLO-based detector enhanced with **time–frequency attention (TF-Attn)** and **multi-resolution spectrograms** for **Low Probability of Intercept (LPI) radar signal detection** in congested spectral environments.

---

## 🔹 Motivation

Conventional spectrogram analysis suffers from the **time–frequency trade-off**:  

- **Short windows** → good time resolution, poor frequency resolution.  
- **Long windows** → good frequency resolution, blurred time localization.  
This trade-off limits radar signal detection in real-world congested environments.  

---
## 🔹 Our Approach

**TF-Attn-YOLO** overcomes this limitation by:  
1. Adding **lightweight time–frequency attention blocks** that capture spectro-temporal dependencies.  
2. Using **multi-resolution spectrogram inputs** to mitigate the Heisenberg uncertainty trade-off.  
3. Enabling both **detection and characterization** of radar signals, including:  
   - Time of arrival  
   - Duration  
   - Bandwidth  
   - Carrier frequency  
---

## 🔹 Features

- ✅ Multi-resolution spectrogram processing  
- ✅ TF-Attn blocks integrated into YOLO backbone/head  
- ✅ Support for **uniresolution** and **multi-resolution** datasets (`.pt` tensors)  
- ✅ Minimal training & prediction scripts (`examples/`)  
- ✅ Dataset generation utilities (`dataset/`)  

---

## 🔹 Project Structure

```
TF-Attn-YOLO-ICASSP/
 │
 ├── dataset/                # Dataset generation, STFT, signals, scenarios
 ├── examples/               # Minimal training, prediction & dataset scripts
 ├── tf_attn_yolo/           # Model, layers, utils (datasets, post-process)
 └── requirements.txt        # Dependencies
```

---

## 🔹 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate a dataset
```bash
python examples/minimal_dataset_generation.py --out ./output
```

### 3. Train TF-Attn-YOLO
```bash
python examples/minimal_train.py \
  --data_dir ./output \
  --dataset multires \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-3 \
  --device auto
```

### 4. Run inference
```bash
python examples/minimal_predict.py --weights ./outputs/minimal_train/best.pt
```

---

## 🔹 Citation

If you use this repository, please cite our **ICASSP 2026** paper:

```bibtex
@inproceedings{mazouz2025tfattnyolo,
  title     = {Multi-Resolution Spectrograms Detection of LPI RADAR with Time-Frequency Attention augmented YOLO},
  author    = {Mazouz, Reihan and Taylor, Abigael and Picheral, José and Bosse, Jonathan and Marcos, Sylvie and Belafdil, Chakib},
  booktitle = {ICASSP},
  year      = {2026}
}
```

---

## 🔹 License
MIT License. See [LICENSE](LICENSE) for details.  
