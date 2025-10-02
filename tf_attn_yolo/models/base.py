import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from torch.cuda.amp import GradScaler, autocast
from typing import Union, Optional, Callable


from ..utils.dataset import YoloPTDataset, YOLODatasetMultiRes
from ..utils.display_outputs import plot_batch_with_boxes
from ..utils.post_process import non_max_suppression 
from ..utils.tal import make_anchors, dist2bbox

def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return [ _to_device(o, device) for o in obj ]
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj

class BaseModel(nn.Module):
    def __init__(self, device="cuda:0", output_dir="outputs"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")
        self.to(self.device)

        self.name = self.__class__.__name__
        self.history = []

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_model_summary(self, model, output_dir, filename="model_summary.txt"):
        """
        Sauvegarde un résumé du modèle dans un fichier texte.
        - model : le modèle PyTorch (hérite de nn.Module)
        - output_dir : dossier où enregistrer le résumé
        - input_shapes : liste des shapes des entrées simulées
        - filename : nom du fichier à écrire
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)

        try:
            # Génère dummy input(s)
            if self.input_resolutions is not None:
                input_shapes = self.input_resolutions
            else: 
                input_shapes = [(1024,1024)]
            dummy_input = [torch.randn(shape).to(model.device) for shape in self.input_resolutions]
            input_data = dummy_input if len(dummy_input) > 1 else dummy_input[0]

            # Résumé structuré avec torchinfo
            model_summary = summary(
                model,
                input_data=input_data,
                depth=3,
                col_names=("input_size", "output_size", "num_params"),
                verbose=0
            )

            with open(path, "w") as f:
                f.write(f"# Model: {model.__class__.__name__}\n")
                f.write(f"# Device: {model.device}\n")
                f.write(str(model_summary))

            print(f"[📄] Model summary saved to {path}")

        except Exception as e:
            # Fallback : print str(model)
            with open(path, "w") as f:
                f.write(f"# Model: {model.__class__.__name__}\n")
                f.write(f"# Device: {model.device}\n")
                f.write(str(model))
                f.write(f"\n\n⚠️ torchinfo.summary failed: {e}")

            print(f"[⚠] Fallback model summary saved to {path}")

    def fit(
        self,
        data_dir: str,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 5,
        dataset: Union[str, type] = "fused",
        use_amp: bool = True,
        evaluator: Optional[Callable] = None,
        monitor: str = "val_loss",
        mode: str = "min",
    ):


        # ---------------- datasets & loaders ----------------
        DATASETS = {
            "mutltires": YOLODatasetMultiRes,
            "unires": YoloPTDataset,
        }
        YOLODataset = DATASETS.get(dataset.lower(), None) if isinstance(dataset, str) else dataset
        if YOLODataset is None:
            raise ValueError(f"Unknown dataset type '{dataset}'")

        train_ds = YOLODataset(os.path.join(data_dir, "train/data"),
                            os.path.join(data_dir, "train/labels"))
        val_ds   = YOLODataset(os.path.join(data_dir, "val/data"),
                            os.path.join(data_dir, "val/labels"))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                pin_memory=True, collate_fn=train_ds.collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                pin_memory=True, collate_fn=val_ds.collate_fn)

        # ---------------- optim, loss, misc -----------------
        os.makedirs(self.output_dir, exist_ok=True)
        best_path = os.path.join(self.output_dir, "best.pt")
        last_path = os.path.join(self.output_dir, "last.pt")

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scaler = GradScaler(enabled=use_amp)
        criterion = self.criterion

        def _to_device_imgs(imgs):
            if isinstance(imgs, list):
                return [img.to(self.device, non_blocking=True) for img in imgs]
            return imgs.to(self.device, non_blocking=True)

        def _build_batch(targets):
            targets = _to_device(targets, self.device)
            return {
                "batch_idx": targets[:, 0].long(),
                "cls":      targets[:, 1].unsqueeze(1).long(),
                "bboxes":   targets[:, 2:6],
                "snr":      targets[:, 6].unsqueeze(1),
            }

        def _forward_loss(imgs, targets):
            with autocast(enabled=use_amp):
                dist_out, clsobj_out = self(imgs)
                feats = dist_out
                pred_scores = torch.cat([x.flatten(2).permute(0, 2, 1) for x in clsobj_out], dim=1)
                pred_distri = torch.cat([x.flatten(2).permute(0, 2, 1) for x in dist_out],  dim=1)
                loss, loss_dict, _ = criterion(pred_distri, pred_scores, targets, feats=feats)
            return loss, loss_dict

        # ---------------- monitoring helpers ----------------
        def _is_better(curr, best):
            return (curr < best) if mode == "min" else (curr > best)

        best_score = float("inf") if mode == "min" else -float("inf")
        no_improve = 0
        history = []

        self.save_model_summary(self, self.output_dir)

        # ===================== training loop =====================
        for epoch in range(1, epochs + 1):
            self.train()
            train_loss_sum = 0.0
            n_train = 0

            for imgs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch} • train", unit="batch", leave=False):
                imgs = _to_device_imgs(imgs)
                batch = _build_batch(targets)

                optimizer.zero_grad(set_to_none=True)
                loss, _ = _forward_loss(imgs, batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += float(loss.item())
                n_train += 1

            train_loss = train_loss_sum / max(1, n_train)

            # --------------------- validation ---------------------
            self.eval()
            val_loss_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for imgs, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch} • val", unit="batch", leave=False):
                    imgs = _to_device_imgs(imgs)
                    batch = _build_batch(targets)
                    loss, _ = _forward_loss(imgs, batch)
                    val_loss_sum += float(loss.item())
                    n_val += 1
            val_loss = val_loss_sum / max(1, n_val)

            metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}

            # Évaluateur optionnel (ex: mAP)
            if evaluator is not None:
                try:
                    extra = evaluator(self, val_loader) or {}
                    metrics.update(extra)
                except Exception as e:
                    print(f"[warn] evaluator failed: {e}")

            # checkpoint last
            torch.save(self.state_dict(), last_path)

            # early stopping & best
            score = metrics.get(monitor, val_loss)
            if _is_better(score, best_score):
                best_score = score
                no_improve = 0
                torch.save(self.state_dict(), best_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[early stop] no improvement in '{monitor}' for {patience} epochs.")
                    history.append(metrics)
                    break

            history.append(metrics)

        return history

    def predict(self, image_tensor, to_plot=False, conf_threshold=0.1, labels=None):
        self.eval()
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.device) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            dist_out, clsobj_out = self(image_tensor)
            feats = dist_out
            processed_output = self.postprocess(dist_out, clsobj_out, feats, conf_thres=conf_threshold)

            if to_plot:
                processed_targets = []
                for pred in processed_output:
                    if pred is not None and len(pred) > 0:
                        boxes = pred[:, [5, 0, 1, 2, 3]]  # cls, x1, y1, x2, y2
                    else:
                        boxes = torch.zeros((0, 5))
                    processed_targets.append(boxes)

                # 💡 Ajout de `labels` ici
                plot_batch_with_boxes(
                    feats=image_tensor,  
                    targets=processed_targets, 
                    class_names=getattr(self, 'class_names', None),
                    save_path=to_plot,
                    max_batch_size=3,
                    labels=labels  
                )

            return processed_output, dist_out, clsobj_out

    def postprocess(self, dist_out, cls_out, feats, conf_thres=0.1, iou_thres=0.1, without_nms=False):
        """
        Postprocessing like YOLOv11 without objectness, with NMS.
        """
        # dist_out et cls_out sont des listes de (B, C, H, W)
        pred_dist = torch.cat([x.flatten(2) for x in dist_out], dim=2).permute(0, 2, 1)  # (B, N, 4*reg_max)
        pred_cls  = torch.cat([x.flatten(2) for x in cls_out],  dim=2).permute(0, 2, 1)  # (B, N, C)
        B, N, _ = pred_dist.shape

        # (3) Anchors
        anchor_points, stride_tensor = make_anchors(feats, self.strides)
        anchor_points = anchor_points.to(pred_dist.device)
        stride_tensor = stride_tensor.to(pred_dist.device)

        # (4) DFL Projection — ⚠️ Cast `proj` to same dtype & device as `pred_dist`
        proj = torch.arange(self.reg_max, dtype=torch.float, device=pred_dist.device)
        proj = proj.to(dtype=pred_dist.dtype)  # AMP compatibility
        pred_ltrb = pred_dist.view(B, N, 4, self.reg_max).softmax(3).matmul(proj)

        # (5) Convertir les distances en boîtes
        pred_bboxes = dist2bbox(pred_ltrb, anchor_points, xywh=False)  # (B, N, 4)
        pred_bboxes_abs = pred_bboxes * stride_tensor  # (B, N, 4)

        # (6) Score des classes
        cls_scores = pred_cls.sigmoid()  # (B, N, C)

        # Convertir xyxy → xywh avant concat
        x1y1 = pred_bboxes_abs[..., :2]
        x2y2 = pred_bboxes_abs[..., 2:4]
        xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        pred_bboxes_xywh = torch.cat([xy, wh], dim=-1)  # (B, N, 4)

        # (7) Empilement [x, y, w, h, conf1, conf2, ..., confN]
        pred_final = torch.cat([pred_bboxes_xywh, cls_scores], dim=2)  # (B, N, 4 + C)
        prediction = pred_final.permute(0, 2, 1)  # (B, 4+C, N)

        if without_nms:
            return prediction

        # (8) NMS
        results = non_max_suppression(
            prediction=prediction,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nc=self.num_classes,
            in_place=True,
            multi_label=True
        )

        return results

    def load_weights(self, weights_path: str, device: str = "cpu", eval_mode: bool = True):
        """
        Charge des poids depuis un fichier .pth/.pt en supprimant les clés inutiles.

        Args:
            weights_path (str): chemin vers le fichier de poids sauvegardé.
            device (str): 'cpu' ou 'cuda'.
            eval_mode (bool): si True, met le modèle en mode eval après chargement.

        Returns:
            missing_keys (list): clés attendues mais manquantes.
            unexpected_keys (list): clés présentes dans les poids mais non utilisées.
        """
        # Charger le dictionnaire brut
        state_dict = torch.load(weights_path, map_location=device)

        # Filtrer pour garder uniquement les clés correspondant au modèle
        clean_state_dict = {
            k: v for k, v in state_dict.items() if k in self.state_dict()
        }

        # Charger avec reporting des clés manquantes/inattendues
        missing_keys, unexpected_keys = self.load_state_dict(clean_state_dict, strict=False)

        # Si demandé, passer en mode eval
        if eval_mode:
            self.eval()
