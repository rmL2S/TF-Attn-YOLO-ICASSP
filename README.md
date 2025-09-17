# TF-Attn-YOLO-ICASSP

Official implementation of **TF-Attn-YOLO**, a YOLO-based detector with **time–frequency attention (TF-Attn)** and **multi-resolution spectrograms** for **Low Probability of Intercept (LPI) radar signal detection** in congested spectral environments.

---

## 🔹 Overview

Conventional spectrograms face a time–frequency trade-off:  
- Short windows → accurate timing but poor frequency resolution.  
- Long windows → better spectral resolution but blurred timing.  

**TF-Attn-YOLO** addresses this limitation by:  
1. Introducing lightweight **time–frequency attention (TF-Attn)** blocks that enhance spectro-temporal patterns.  
2. Leveraging **multi-resolution spectrograms** processed jointly, mitigating the Heisenberg trade-off.  
3. Providing both **signal detection and characterization** (time of arrival, duration, bandwidth, carrier frequency).


