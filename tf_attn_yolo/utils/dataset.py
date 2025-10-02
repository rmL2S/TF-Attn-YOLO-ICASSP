import os, re, json
from pathlib import Path
from typing import List, Tuple, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_NUMERIC_RE = re.compile(r"(\d+)$")

class YOLODatasetMultiRes(Dataset):
    """
    Multi-resolution dataset for spectrograms saved as .pt files (one list per sample)
    with label metadata stored as JSON.

    Each .pt file must contain a Python list of Tensors shaped [C, H, W] (or [H, W]),
    one tensor per resolution. The corresponding JSON file (same stem) must live in
    `labels_dir` and contain a list of objects with at least:
        {
          "class": <int or float>,
          "xc": <float>, "yc": <float>, "w": <float>, "h": <float>,  # normalized [0,1]
          "snr": <float, optional>,
          "psnr": { "cfg128": <float>, "cfg256": <float>, ... }      # optional per-res keys
        }

    Output of __getitem__:
        {
          "imgs":    List[Tensor]  # len R, each FloatTensor [C, H, W]
          "cls":     FloatTensor [N]
          "bboxes":  FloatTensor [N, 4]  # normalized (xc, yc, w, h)
          "snr":     FloatTensor [N]
          "psnr":    FloatTensor [N, R]  # -1.0 where missing
          "img_idx": int
          "res_keys": List[str] | None
        }

    Collate output:
        imgs:   List[FloatTensor], length R, each [B, C, H, W]
        targets: FloatTensor [M, 7 + R] with columns:
                 [img_idx, cls, xc, yc, w, h, snr, psnr_0, ..., psnr_{R-1}]
        res_keys: List[str] | None
    """

    def __init__(
        self,
        data_dir: str,
        labels_dir: str,
        res_keys: Optional[Sequence[str]] = ("cfg128", "cfg256", "cfg512", "cfg1024", "cfg2048"),
        max_dim: int = 1024,
    ) -> None:
        self.data_paths: List[str] = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".pt")
        )
        if not self.data_paths:
            raise FileNotFoundError(f"No .pt files found in: {data_dir}")

        self.labels_dir = labels_dir
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(f"Labels dir not found: {self.labels_dir}")

        self._res_keys: Optional[List[str]] = list(res_keys) if res_keys is not None else None
        self.max_dim = int(max_dim)

    def __len__(self) -> int:
        return len(self.data_paths)

    @staticmethod
    def _numeric_key(k: str) -> int:
        m = _NUMERIC_RE.search(k)
        return int(m.group(1)) if m else 10**9  # push non-numeric to the end

    def _ensure_res_keys(self, psnr_dict: dict) -> None:
        if self._res_keys is not None:
            return
        if not isinstance(psnr_dict, dict) or not psnr_dict:
            return
        keys = list(psnr_dict.keys())
        keys.sort(key=self._numeric_key)
        self._res_keys = keys

    @staticmethod
    def _to_chw(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            return t.unsqueeze(0).contiguous()
        if t.ndim == 3:
            return t.contiguous()
        raise ValueError(f"Unexpected tensor shape: {tuple(t.shape)} (expected 2D or 3D)")

    def __getitem__(self, idx: int) -> dict:
        data_path = self.data_paths[idx]
        specs = torch.load(data_path, map_location="cpu")

        if not isinstance(specs, (list, tuple)):
            raise TypeError(f"{os.path.basename(data_path)}: expected a list/tuple of Tensors, got {type(specs)}")

        imgs: List[torch.Tensor] = []
        for t in specs:
            if not torch.is_tensor(t):
                raise TypeError(f"{os.path.basename(data_path)}: list contains non-tensor element of type {type(t)}")
            t = self._to_chw(t)
            _, H, W = t.shape
            if H <= self.max_dim and W <= self.max_dim:
                imgs.append(t.to(torch.float32))

        if not imgs:
            raise ValueError(f"{os.path.basename(data_path)} produced no valid resolutions under max_dim={self.max_dim}")

        base = os.path.splitext(os.path.basename(data_path))[0]
        json_path = os.path.join(self.labels_dir, base + ".json")

        cls: List[float] = []
        bboxes: List[List[float]] = []
        snrs: List[float] = []
        psnrs_per_obj: List[List[float]] = []

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                meta = json.load(f)

            if self._res_keys is None:
                for item in meta.get("labels", []):
                    if isinstance(item.get("psnr"), dict) and item["psnr"]:
                        self._ensure_res_keys(item["psnr"])
                        break

            for item in meta.get("labels", []):
                cls.append(float(item["class"]))
                bboxes.append([float(item["xc"]), float(item["yc"]), float(item["w"]), float(item["h"])])
                snr_val = item.get("snr", None)
                snrs.append(float(snr_val) if snr_val is not None else -1.0)

                vec: List[float] = []
                if self._res_keys is None:
                    if isinstance(item.get("psnr"), dict) and item["psnr"]:
                        local_keys = sorted(item["psnr"].keys(), key=self._numeric_key)
                        vec = [float(item["psnr"].get(k, -1.0)) for k in local_keys]
                        self._res_keys = local_keys
                    else:
                        vec = []
                else:
                    for k in self._res_keys:
                        v = item.get("psnr", {}).get(k, None)
                        vec.append(float(v) if v is not None else -1.0)

                psnrs_per_obj.append(vec)

        cls_t = torch.tensor(cls, dtype=torch.float32)
        bboxes_t = (
            torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4), dtype=torch.float32)
        )
        snrs_t = torch.tensor(snrs, dtype=torch.float32)

        R = len(self._res_keys) if self._res_keys is not None else 0
        if len(psnrs_per_obj) == 0:
            psnr_t = torch.zeros((0, R), dtype=torch.float32)
        else:
            if R == 0:
                R = len(psnrs_per_obj[0]) if psnrs_per_obj else 0
                self._res_keys = [f"cfg{i}" for i in range(R)]
            psnr_t = torch.tensor(psnrs_per_obj, dtype=torch.float32)
            if psnr_t.ndim != 2 or psnr_t.shape[1] != R:
                N = psnr_t.shape[0]
                fixed = torch.full((N, R), -1.0, dtype=torch.float32)
                cols = min(R, psnr_t.shape[1]) if psnr_t.ndim == 2 else 0
                if cols > 0:
                    fixed[:, :cols] = psnr_t[:, :cols]
                psnr_t = fixed

        return {
            "imgs": imgs,
            "cls": cls_t,
            "bboxes": bboxes_t,
            "snr": snrs_t,
            "psnr": psnr_t,
            "img_idx": idx,
            "res_keys": None if self._res_keys is None else list(self._res_keys),
        }

    @staticmethod
    def collate_fn(batch: List[dict]) -> Tuple[List[torch.Tensor], torch.Tensor, Optional[List[str]]]:
        if not batch:
            return [], torch.zeros((0, 7), dtype=torch.float32), None

        num_resolutions = [len(item["imgs"]) for item in batch]
        if len(set(num_resolutions)) != 1:
            raise ValueError(f"Inconsistent number of resolutions across batch: {num_resolutions}")
        R = num_resolutions[0]

        imgs_lists: List[List[torch.Tensor]] = [item["imgs"] for item in batch]
        imgs_per_res = list(zip(*imgs_lists))  # length R, tuples of [B] tensors

        imgs: List[torch.Tensor] = []
        for r, res_list in enumerate(imgs_per_res):
            shapes = {tuple(t.shape) for t in res_list}
            if len(shapes) != 1:
                raise ValueError(f"Resolution #{r} has non-matching shapes across batch: {shapes}")
            imgs.append(torch.stack(res_list, dim=0))  # [B, C, H, W]

        all_cls = [item["cls"] for item in batch]
        all_boxes = [item["bboxes"] for item in batch]
        all_snrs = [item["snr"] for item in batch]
        all_psnrs = [item["psnr"] for item in batch]

        targets_rows: List[torch.Tensor] = []
        for i, (cls_t, boxes_t, snr_t, psnr_t) in enumerate(zip(all_cls, all_boxes, all_snrs, all_psnrs)):
            if boxes_t.numel():
                if psnr_t.ndim != 2 or psnr_t.shape[0] != boxes_t.shape[0]:
                    raise ValueError(
                        f"PSNR tensor shape {tuple(psnr_t.shape)} is incompatible with boxes {tuple(boxes_t.shape)}"
                    )
                img_idx = torch.full((boxes_t.shape[0], 1), float(i), dtype=torch.float32)
                cls_col = cls_t.unsqueeze(-1)
                snr_col = snr_t.unsqueeze(-1)
                row = torch.cat((img_idx, cls_col, boxes_t, snr_col, psnr_t), dim=1)
                targets_rows.append(row)

        psnr_cols = all_psnrs[0].shape[1] if all_psnrs and all_psnrs[0].ndim == 2 else 0
        targets = (
            torch.cat(targets_rows, dim=0)
            if targets_rows
            else torch.zeros((0, 7 + psnr_cols), dtype=torch.float32)
        )

        res_keys: Optional[List[str]] = None
        for item in batch:
            if item.get("res_keys"):
                res_keys = item["res_keys"]
                break

        return imgs, targets, res_keys



class YoloPTDataset(Dataset):
    """
    Dataset for spectrogram tensors saved as .pt and YOLO-format labels (.txt).

    Inputs
    ------
    images_dir : str
        Directory containing .pt tensors shaped (H, W) or (C, H, W).
    labels_dir : str
        Directory containing YOLO label files (.txt) with lines: "class cx cy w h" (normalized).
    target_size : int
        Output square size (T x T). Images are center-padded/cropped to this size.
    pad_value : float
        Constant value used for padding.
    snr_fill : float
        Default SNR value to attach to targets when not provided by labels.
    psnr_fill : float
        Default PSNR value to attach to targets when not provided by labels.

    __getitem__ output
    ------------------
    {
      "img":     FloatTensor [C, T, T],
      "targets": FloatTensor [N, 8]  where columns are:
                 [img_idx, cls, cx, cy, w, h, snr, psnr]
                 (cx, cy, w, h are normalized in [0, 1] w.r.t. T)
    }

    Notes
    -----
    - Boxes are reprojected to account for center pad/crop.
    - If a label file is missing or produces no valid boxes, targets will be shape (0, 8).
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        target_size: int = 1024,
        pad_value: float = 0.0,
        snr_fill: float = 0.0,
        psnr_fill: float = -1.0,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Labels dir not found: {self.labels_dir}")

        self.paths: List[Path] = sorted(self.images_dir.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files found in {self.images_dir}")

        self.T = int(target_size)
        self.pad_value = float(pad_value)
        self.snr_fill = float(snr_fill)
        self.psnr_fill = float(psnr_fill)

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _center_pad_crop_2d(
        x: torch.Tensor, out_h: int, out_w: int, pad_value: float
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Center pad to at least (out_h, out_w) then center-crop to exact size.
        x is [C, H, W]. Returns (y, top_pad, left_pad, crop_top, crop_left).
        """
        _, H, W = x.shape
        dh = out_h - H
        dw = out_w - W

        top_pad = max(dh // 2, 0)
        bottom_pad = max(dh - dh // 2, 0)
        left_pad = max(dw // 2, 0)
        right_pad = max(dw - dw // 2, 0)

        y = x
        if top_pad or bottom_pad or left_pad or right_pad:
            # F.pad pads last two dims in order (left, right, top, bottom)
            y = F.pad(y, (left_pad, right_pad, top_pad, bottom_pad), value=pad_value)

        crop_top = max((-dh) // 2, 0)
        crop_left = max((-dw) // 2, 0)
        crop_bottom = crop_top + out_h
        crop_right = crop_left + out_w
        if crop_top or crop_left:
            y = y[:, crop_top:crop_bottom, crop_left:crop_right]

        return y, top_pad, left_pad, crop_top, crop_left

    def _read_labels_txt(self, stem: str) -> List[List[float]]:
        """
        Read YOLO labels for given file stem. Returns a list of [cls, cx, cy, w, h] (normalized).
        Skips malformed lines.
        """
        txt_path = self.labels_dir / f"{stem}.txt"
        if not txt_path.exists():
            return []

        raw = txt_path.read_text().splitlines()
        out: List[List[float]] = []
        for line in raw:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                c, cx, cy, w, h = map(float, parts)
            except ValueError:
                continue
            out.append([c, cx, cy, w, h])
        return out

    def __getitem__(self, idx: int) -> dict:
        p = self.paths[idx]
        stem = p.stem

        # Load .pt -> tensor [C, H, W] or [H, W]
        arr = torch.load(p, map_location="cpu")
        if isinstance(arr, dict) and "image" in arr:
            arr = arr["image"]
        if not torch.is_tensor(arr):
            raise TypeError(f"{p.name}: loaded object is not a torch.Tensor")

        if arr.ndim == 2:
            arr = arr.unsqueeze(0)  # [1, H, W]
        elif arr.ndim != 3:
            raise ValueError(f"{p.name}: expected 2D or 3D tensor, got shape {tuple(arr.shape)}")

        _, H, W = arr.shape

        # Center pad/crop to [C, T, T]
        img, top_pad, left_pad, crop_top, crop_left = self._center_pad_crop_2d(
            arr, self.T, self.T, self.pad_value
        )

        # Read labels and reproject to the new coordinate frame (after pad/crop)
        raw_labels = self._read_labels_txt(stem)
        targets_list: List[List[float]] = []
        for c, cx_n, cy_n, w_n, h_n in raw_labels:
            cx_abs = cx_n * W
            cy_abs = cy_n * H
            w_abs = w_n * W
            h_abs = h_n * H

            cx_adj = cx_abs + left_pad - crop_left
            cy_adj = cy_abs + top_pad - crop_top

            x1 = cx_adj - w_abs / 2.0
            y1 = cy_adj - h_abs / 2.0
            x2 = cx_adj + w_abs / 2.0
            y2 = cy_adj + h_abs / 2.0

            x1 = max(0.0, min(float(self.T), float(x1)))
            y1 = max(0.0, min(float(self.T), float(y1)))
            x2 = max(0.0, min(float(self.T), float(x2)))
            y2 = max(0.0, min(float(self.T), float(y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            cx_new = ((x1 + x2) * 0.5) / self.T
            cy_new = ((y1 + y2) * 0.5) / self.T
            w_new = (x2 - x1) / self.T
            h_new = (y2 - y1) / self.T

            targets_list.append(
                [0.0, float(c), cx_new, cy_new, w_new, h_new, self.snr_fill, self.psnr_fill]
            )

        img = img.to(torch.float32)
        if targets_list:
            targets = torch.tensor(targets_list, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 8), dtype=torch.float32)

        return {"img": img, "targets": targets}

    @staticmethod
    def collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Collate into (imgs, targets, res_keys).
        - imgs:    FloatTensor [B, C, T, T]
        - targets: FloatTensor [M, 8] with img indices set to [0..B-1]
        - res_keys: empty list for API compatibility
        """
        imgs = torch.stack([b["img"] for b in batch], dim=0)
        all_tgts: List[torch.Tensor] = []
        for i, b in enumerate(batch):
            t = b["targets"]
            if t.numel():
                t = t.clone()
                t[:, 0] = i
                all_tgts.append(t)
        targets = torch.cat(all_tgts, dim=0) if all_tgts else torch.zeros((0, 8), dtype=torch.float32)
        return imgs, targets, []