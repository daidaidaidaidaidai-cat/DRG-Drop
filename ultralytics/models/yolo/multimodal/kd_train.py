"""Knowledge-distillation trainer for multimodal YOLO detection."""

from __future__ import annotations

import types
from copy import copy

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.multimodal.train import MultiModalDetectionTrainer
from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import LOGGER


def _iter_matching_tensors(student, teacher):
    if isinstance(student, torch.Tensor) and isinstance(teacher, torch.Tensor):
        yield student, teacher
        return
    if isinstance(student, dict) and isinstance(teacher, dict):
        shared = [k for k in student.keys() if k in teacher]
        if not shared:
            raise ValueError("student/teacher prediction dicts do not share any keys")
        for key in shared:
            yield from _iter_matching_tensors(student[key], teacher[key])
        return
    if isinstance(student, (list, tuple)) and isinstance(teacher, (list, tuple)):
        if len(student) != len(teacher):
            raise ValueError(f"student/teacher prediction lengths differ: {len(student)} vs {len(teacher)}")
        for s_item, t_item in zip(student, teacher):
            yield from _iter_matching_tensors(s_item, t_item)
        return
    raise ValueError(f"unsupported distillation payloads: {type(student)} vs {type(teacher)}")


def _raw_teacher_preds(teacher_model: torch.nn.Module, images: torch.Tensor):
    teacher_model.eval()
    with torch.no_grad():
        teacher_out = teacher_model(images)
    if isinstance(teacher_out, tuple) and len(teacher_out) == 2:
        return teacher_out[1]
    return teacher_out


def _kd_loss(student_preds, teacher_preds, reg_channels: int, nc: int, temperature: float) -> torch.Tensor:
    losses = []
    for s_tensor, t_tensor in _iter_matching_tensors(student_preds, teacher_preds):
        if s_tensor.shape != t_tensor.shape:
            raise ValueError(f"teacher/student prediction shapes differ: {tuple(s_tensor.shape)} vs {tuple(t_tensor.shape)}")
        t_tensor = t_tensor.detach()
        if s_tensor.ndim < 4 or s_tensor.shape[1] < reg_channels + nc:
            losses.append(F.mse_loss(s_tensor, t_tensor))
            continue

        s_box, s_cls = s_tensor.split((reg_channels, nc), dim=1)
        t_box, t_cls = t_tensor.split((reg_channels, nc), dim=1)
        box_loss = F.smooth_l1_loss(s_box, t_box)

        s_logits = (s_cls / temperature).flatten(2).transpose(1, 2)
        t_logits = (t_cls / temperature).flatten(2).transpose(1, 2)
        cls_loss = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction="batchmean",
        ) * (temperature**2)
        losses.append(box_loss + cls_loss)

    if not losses:
        raise ValueError("no matching prediction tensors found for KD")
    return torch.stack([loss.float() for loss in losses]).mean()


class MultiModalKDDetectionTrainer(MultiModalDetectionTrainer):
    """Multi-modal detection trainer with optional compatible-teacher KD."""

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "kd_loss") if self._kd_requested() else ("box_loss", "cls_loss", "dfl_loss")
        return MultiModalDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        teacher_weights = getattr(self.args, "teacher_weights", None)
        if not teacher_weights:
            return model

        teacher_model, _ = attempt_load_one_weight(teacher_weights, device="cpu", fuse=False)
        teacher_model.requires_grad_(False)

        kd_alpha = float(getattr(self.args, "kd_alpha", 0.25))
        kd_temperature = max(float(getattr(self.args, "kd_temperature", 1.0)), 1e-6)
        head = model.model[-1]
        reg_channels = int(getattr(head, "reg_max", 16) * 4)
        nc = int(getattr(head, "nc", self.data["nc"]))
        original_loss = model.loss

        def loss_with_kd(this_model, batch, preds=None):
            if preds is None:
                preds = this_model.forward(batch["img"])
            det_total, det_items = original_loss(batch, preds=preds)
            teacher_preds = _raw_teacher_preds(teacher_model.to(batch["img"].device), batch["img"])
            kd_value = _kd_loss(preds, teacher_preds, reg_channels=reg_channels, nc=nc, temperature=kd_temperature)
            total_loss = det_total + kd_alpha * kd_value
            kd_item = (kd_alpha * kd_value.detach()).reshape(1)
            if torch.is_tensor(det_items):
                det_items = torch.cat([det_items, kd_item], dim=0)
            return total_loss, det_items

        model.loss = types.MethodType(loss_with_kd, model)
        LOGGER.info(
            f"Enabled multimodal KD: teacher={teacher_weights}, alpha={kd_alpha:.3f}, temperature={kd_temperature:.3f}"
        )
        LOGGER.info("KD expects teacher/student to share compatible raw detection output shapes, e.g. same P2 four-scale head.")
        return model

    def _kd_requested(self) -> bool:
        teacher_weights = getattr(self.args, "teacher_weights", None)
        return bool(teacher_weights)
