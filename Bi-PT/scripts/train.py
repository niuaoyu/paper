#!/usr/bin/env python3
"""Bi-PT training entry point.

Trains the bidirectional cross-attention model (``DeformBlockConcat``) or the
single-cross-attention ablation (``DeformBlockConcatSingleCA``) with the
semantic-aware Chamfer loss (deep-supervised on both NODE outputs) plus a
Laplacian regularization. See the README for paper-matching commands and the
Table-2 ablation flags.
"""
import os
import sys
import glob
import random
import argparse
import copy
from contextlib import contextmanager
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing

# Allow running as `python scripts/train.py` from the repo root without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bipt.models import DeformBlockConcat, DeformBlockConcatSingleCA
from bipt.losses import chamfer_distance
from bipt.data.dataset import (
    LabeledDataset,
    collate_batch,
    load_vtp_points_and_labels,
    load_vtp_faces,
    build_atlas_feat,
    build_atlas_feat_bn6,
)


# ----------------------------- device -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chamber label IDs (RA/LA/RV/LV/MYO)
LABEL_IDS: List[int] = [2, 3, 4, 5, 6]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_rng_state():
    state = {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    random.setstate(state["py_random"])
    np.random.set_state(state["np_random"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def make_autocast_and_scaler(use_amp: bool, device_: torch.device):
    """
    Returns (autocast_ctx, scaler, amp_enabled_runtime).
    """
    if use_amp and device_.type == "cuda":
        from torch.cuda.amp import autocast as cuda_autocast, GradScaler
        return cuda_autocast, GradScaler(enabled=True), True

    @contextmanager
    def noop_autocast(enabled=True):
        yield

    from torch.cuda.amp import GradScaler
    return noop_autocast, GradScaler(enabled=False), False


def cd_plain(x, y) -> torch.Tensor:
    loss, _ = chamfer_distance(
        x, y,
        batch_reduction="mean",
        point_reduction="mean",
        norm=2,
        single_directional=False,
        x_labels=None,
        y_labels=None,
    )
    return loss.float()


def cd_with_labels(x, y, x_labels, y_labels, wx: float, wy: float) -> torch.Tensor:
    loss, _ = chamfer_distance(
        x, y,
        batch_reduction="mean",
        point_reduction="mean",
        norm=2,
        single_directional=False,
        x_labels=x_labels,
        y_labels=y_labels,
        weight_x=float(wx),
        weight_y=float(wy),
    )
    return loss.float()


def compute_loss_labeldim2(pred, gt_xyz, pred_pairs, gt_pairs, cd_mode: str, wx: float, wy: float) -> torch.Tensor:
    """
    label_dim=2 => original behavior: plain / pair-label / both
    """
    if cd_mode == "plain":
        return cd_plain(pred, gt_xyz)
    if cd_mode == "label":
        return cd_with_labels(pred, gt_xyz, pred_pairs, gt_pairs, wx, wy)
    if cd_mode == "both":
        return 0.5 * (cd_plain(pred, gt_xyz) + cd_with_labels(pred, gt_xyz, pred_pairs, gt_pairs, wx, wy))
    raise ValueError("cd_mode must be one of: plain | label | both")


def compute_loss_labeldim3_three_terms(
    pred,
    gt_xyz,
    pred_pairs,
    gt_pairs,
    pred_lab3,
    gt_lab3,
    wx: float,
    wy: float,
    w_plain: float,
    w_pair2: float,
    w_label3: float,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    label_dim=3 => THREE-TERM loss ALWAYS:
      1) plain
      2) pair2
      3) label3
    """
    l0 = cd_plain(pred, gt_xyz)
    l2 = cd_with_labels(pred, gt_xyz, pred_pairs, gt_pairs, wx, wy)
    l3 = cd_with_labels(pred, gt_xyz, pred_lab3, gt_lab3, wx, wy)  # triplet

    w0, w2, w3 = float(w_plain), float(w_pair2), float(w_label3)
    denom = max(1e-12, (w0 + w2 + w3))
    total = (w0 * l0 + w2 * l2 + w3 * l3) / denom
    return total.float(), (l0.float(), l2.float(), l3.float())


def build_pred_idx_by_label(labels: torch.Tensor, label_ids: List[int]) -> Dict[int, torch.Tensor]:
    """
    labels: (N,D) long, D=2 or 3. A vertex belongs to chamber l if ANY component == l.
    Returns dict l -> indices tensor on same device.
    """
    out: Dict[int, torch.Tensor] = {}
    for l in label_ids:
        mask = (labels == int(l)).any(dim=1)
        out[int(l)] = torch.where(mask)[0]
    return out


def label_subset_cd_batch_mean(
    pred_verts: torch.Tensor,         # (B, Nverts, 3) float32
    y_tgt: torch.Tensor,              # (B, Ngt, 3) float32
    y_labels: torch.Tensor,           # (B, Ngt, D) long, D=2 or 3
    pred_idx_by_label: Dict[int, torch.Tensor],
    label_ids: List[int],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Returns:
      label_sum: scalar tensor = sum_l mean_b CD_l(b)
      label_cds: dict l -> scalar tensor mean over batch (divide by B like your earlier logic)
    """
    B = int(pred_verts.shape[0])
    label_sum = pred_verts.new_zeros(())  # scalar
    label_cds: Dict[int, torch.Tensor] = {}

    for l in label_ids:
        pred_idx = pred_idx_by_label.get(int(l), None)
        if pred_idx is None or pred_idx.numel() == 0:
            label_cds[int(l)] = pred_verts.new_zeros(())
            continue

        loss_l_sum = pred_verts.new_zeros(())
        for b in range(B):
            # pred subset: (1,P,3)
            px = pred_verts[b:b+1, pred_idx, :]

            # gt subset indices: (Ngt,) mask
            mask_y = (y_labels[b] == int(l)).any(dim=1)
            y_idx = torch.where(mask_y)[0]
            if y_idx.numel() == 0:
                # no gt points with this label in this sample => contribute 0
                continue
            py = y_tgt[b:b+1, y_idx, :]

            cd_l = cd_plain(px, py)
            loss_l_sum = loss_l_sum + cd_l

        loss_l = loss_l_sum / max(1, B)
        label_cds[int(l)] = loss_l
        label_sum = label_sum + loss_l

    return label_sum, label_cds


def build_faces_by_label_from_lab3(
    faces: torch.Tensor,          # (F,3) long on device
    lab3: torch.Tensor,           # (N,3) long on device
    label_ids: List[int],
) -> Dict[int, Optional[torch.Tensor]]:
    """
    Face belongs to chamber l if ALL 3 vertices' lab3 tuple contains l.
    Returns dict l -> faces_l (F_l,3) or None if empty.
    """
    out: Dict[int, Optional[torch.Tensor]] = {}
    v0 = faces[:, 0]
    v1 = faces[:, 1]
    v2 = faces[:, 2]
    lab_v0 = lab3[v0]  # (F,3)
    lab_v1 = lab3[v1]
    lab_v2 = lab3[v2]

    for l in label_ids:
        l = int(l)
        m0 = (lab_v0 == l).any(dim=1)
        m1 = (lab_v1 == l).any(dim=1)
        m2 = (lab_v2 == l).any(dim=1)
        mf = m0 & m1 & m2
        idx = torch.where(mf)[0]
        if idx.numel() == 0:
            out[l] = None
        else:
            out[l] = faces[idx].contiguous()
    return out


def _looks_like_underflow_assertion(e: AssertionError) -> bool:
    msg = str(e).lower()
    needles = ["underflow", "gradscaler", "foundinf", "inf", "nan", "loss scale", "scale", "fp16", "half"]
    return any(n in msg for n in needles)


def train(args):
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # AUTO log file
    if args.log_file is None or str(args.log_file).strip() == "" or str(args.log_file).lower().strip() == "auto":
        args.log_file = os.path.join(args.ckpt_dir, "train.txt")
    if os.path.dirname(args.log_file):
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    label_dim = int(args.label_dim)
    if label_dim not in (2, 3):
        raise ValueError("--label-dim must be 2 or 3")

    # chamber lap is only meaningful for lab3 faces
    if args.use_chamber_lap and label_dim != 3:
        print("[WARN] --use-chamber-lap requested but label_dim!=3. DISABLING chamber lap.")
        args.use_chamber_lap = False

    spc_feature_dim = 6 if args.no_label_feat else 8
    d_points_spc = spc_feature_dim - 3  # 3 or 5

    # choose dirs/arrays
    if label_dim == 2:
        gt_dir = args.gt_dir_2
        atlas_path = args.atlas_path_2
        gt_pair_array = args.gt_pair_array_2
        atlas_pair_array = args.atlas_pair_array_2
        gt_label3_array = args.gt_label3_array_2
        atlas_label3_array = args.atlas_label3_array_2
    else:
        gt_dir = args.gt_dir_3
        atlas_path = args.atlas_path_3
        gt_pair_array = args.gt_pair_array_3
        atlas_pair_array = args.atlas_pair_array_3
        gt_label3_array = args.gt_label3_array_3
        atlas_label3_array = args.atlas_label3_array_3

    atlas_feature_dim = 6 if args.no_label_feat else (8 if label_dim == 2 else 9)
    d_points_atlas = atlas_feature_dim - 3  # 3 or 5 or 6

    # AMP setup
    autocast_ctx, scaler, amp_runtime = make_autocast_and_scaler(args.amp, device)

    # l1/l2 weights
    w1 = float(args.l1_loss_weight)
    if not (0.0 <= w1 <= 1.0):
        raise ValueError("--l1-loss-weight must be in [0,1]")
    w2 = 1.0 - w1

    header = (
        f"DEVICE={device.type}\n"
        f"AMP={bool(args.amp)} (runtime={amp_runtime}) | ODE_FORCE_FP32={bool(args.ode_force_fp32)}\n"
        f"Model: ca_type={args.ca_type} time={args.time} num_hidden={args.num_hidden} latent_len={args.latent_len}\n"
        f"tol={args.tol} tanh_dynamics={args.tanh_dynamics} ca_nblocks={args.ca_nblocks} ca_nneighbor={args.ca_nneighbor}\n"
        f"affine_dynamics={args.affine_dynamics} cond_mode={args.cond_mode} cond_norm={args.cond_norm} label_dim={label_dim}\n"
        f"SPC: dir={args.sparse_dir} pcs_key={args.sparse_pcs_key} pairs_key={args.sparse_pairs_key} "
        f"feat_dim={spc_feature_dim} d_points_spc={d_points_spc}\n"
        f"GT:  dir={gt_dir} pair_array={gt_pair_array} label3_array={gt_label3_array}\n"
        f"ATLAS: path={atlas_path} pair_array={atlas_pair_array} label3_array={atlas_label3_array} "
        f"feat_dim={atlas_feature_dim} d_points_atlas={d_points_atlas}\n"
        f"LOSS: cd_mode(label_dim=2)={args.cd_mode}, w1={w1}, w2={w2}, lap={args.lap_weight} | cd3_weights(label_dim=3) "
        f"plain={args.cd3_w_plain} pair2={args.cd3_w_pair2} label3={args.cd3_w_label3}\n"
        f"label_wx={args.label_cd_wx} label_wy={args.label_cd_wy} label_scale={args.label_scale}\n"
        f"EXTRA: use_chamber_cd={bool(args.use_chamber_cd)} chamber_cd_w={args.chamber_cd_weight} "
        f"use_chamber_lap={bool(args.use_chamber_lap)} chamber_lap_w={args.chamber_lap_weight} label_ids={LABEL_IDS}\n"
        f"OPT: AdamW lr={args.lr} wd={args.weight_decay} grad_clip={args.grad_clip}\n"
        f"TRAIN: epochs={args.epochs} bs={args.batch_size} workers={args.num_workers} seed={args.seed}\n"
        f"LOG:  {args.log_file}\n"
    )
    print(header)
    with open(args.log_file, "a") as f:
        f.write(header + "\n")

    # dataset/loader
    ds = LabeledDataset(
        sparse_dir=args.sparse_dir,
        gt_dir=gt_dir,
        pcs_key=args.sparse_pcs_key,
        spc_pairs_key=args.sparse_pairs_key,
        gt_pair_array=gt_pair_array,
        gt_label3_array=gt_label3_array,
        label_dim=label_dim,
        label_scale=float(args.label_scale),
        no_label_feat=args.no_label_feat,
    )

    def make_loader(batch_size: int) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_batch,
        )

    loader = make_loader(int(args.batch_size))

    # atlas load: always load pair2 (for pair-CD) + faces
    atlas_xyz, atlas_pairs_np = load_vtp_points_and_labels(atlas_path, atlas_pair_array, min_dim=2)
    atlas_faces_np = load_vtp_faces(atlas_path)

    atlas_pairs = torch.from_numpy(atlas_pairs_np).long().to(device)  # (Na,2)
    atlas_faces = torch.from_numpy(atlas_faces_np).long().to(device)  # (F,3)

    # atlas label3 only if label_dim=3 (for feature BN9 + label3-CD + chamber lap)
    if label_dim == 3:
        _, atlas_lab3_np = load_vtp_points_and_labels(atlas_path, atlas_label3_array, min_dim=3)
        atlas_lab3 = torch.from_numpy(atlas_lab3_np).long().to(device)  # (Na,3)
    else:
        atlas_lab3_np = None
        atlas_lab3 = None

    # build atlas feature q_item:
    if args.no_label_feat:
        q_item_np = build_atlas_feat_bn6(atlas_xyz)  # (Na,6)
    else:
        if label_dim == 2:
            q_item_np = build_atlas_feat(atlas_xyz, atlas_pairs_np, feat_label_dim=2, label_scale=float(args.label_scale))
        else:
            q_item_np = build_atlas_feat(atlas_xyz, atlas_lab3_np, feat_label_dim=3, label_scale=float(args.label_scale))

    q_item = torch.from_numpy(q_item_np).float().to(device)  # (Na,8 or 9 or 6)

    # -------- NEW: precompute pred_idx_by_label for chamber CD --------
    if args.use_chamber_cd:
        if label_dim == 2:
            pred_idx_by_label = build_pred_idx_by_label(atlas_pairs, LABEL_IDS)
        else:
            if atlas_lab3 is None:
                raise RuntimeError("label_dim=3 but atlas_lab3 is None (check atlas_label3_array).")
            pred_idx_by_label = build_pred_idx_by_label(atlas_lab3, LABEL_IDS)
    else:
        pred_idx_by_label = {}

    # -------- NEW: precompute faces_by_label for chamber Lap (dim=3 only) -------
    if args.use_chamber_lap:
        if atlas_lab3 is None:
            raise RuntimeError("--use-chamber-lap requires label_dim=3 and atlas VertexLabels3 present.")
        faces_by_label_torch = build_faces_by_label_from_lab3(atlas_faces, atlas_lab3, LABEL_IDS)
    else:
        faces_by_label_torch = None

    # model
    def _auto_to_none(s):
        return None if (s is None or str(s).lower() == "auto") else str(s).lower()

    ca_dec_fc = _auto_to_none(args.ca_dec_fc_norm_type)
    ca_dec_fp = _auto_to_none(args.ca_dec_fp_norm_type)

    ModelCls = DeformBlockConcat if args.ca_type == "double" else DeformBlockConcatSingleCA

    model = ModelCls(
        time=args.time,
        num_hidden=args.num_hidden,
        latent_len=args.latent_len,
        tol=args.tol,
        tanh_dynamics=args.tanh_dynamics,
        d_points_spc=d_points_spc,
        d_points_atlas=d_points_atlas,
        ca_nblocks=args.ca_nblocks,
        ca_nneighbor=args.ca_nneighbor,
        norm_type=args.norm_type,
        cond_mode=args.cond_mode,
        cond_norm=args.cond_norm,
        ca_dec_fc_norm_type=ca_dec_fc,
        ca_dec_fp_norm_type=ca_dec_fp,
        ode_force_fp32=bool(args.ode_force_fp32),
        affine_dynamics=bool(args.affine_dynamics),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch = 1
    if args.resume is not None and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        opt.load_state_dict(ckpt["opt_state"])
        sched.load_state_dict(ckpt["sched_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        args.grad_clip = float(ckpt.get("grad_clip", args.grad_clip))
        autocast_ctx, scaler, amp_runtime = make_autocast_and_scaler(args.amp, device)
        loader = make_loader(int(args.batch_size))
        print(f"Resuming from epoch {start_epoch}.")

    model.train()

    def _run_one_epoch(epoch: int, lap_w: float):
        sum_total = 0.0
        sum_cd1 = 0.0
        sum_cd2 = 0.0
        sum_lap = 0.0
        n_seen = 0

        # breakdown for label_dim=3
        sum_cd2_plain = 0.0
        sum_cd2_pair2 = 0.0
        sum_cd2_lab3 = 0.0

        # -------- NEW: extra accumulators (per-chamber CD/Lap) --------
        sum_chcd = 0.0
        sum_chcd_by_label = {l: 0.0 for l in LABEL_IDS}

        sum_chlap = 0.0
        sum_chlap_by_label = {l: 0.0 for l in LABEL_IDS}
        # -------------------------------------------------------------

        for it, (spc_feat, gt_xyz, gt_pairs, gt_lab3) in enumerate(loader):
            spc_feat = spc_feat.to(device, non_blocking=True).float()
            gt_xyz = gt_xyz.to(device, non_blocking=True).float()
            gt_pairs = gt_pairs.to(device, non_blocking=True).long()
            if gt_lab3 is not None:
                gt_lab3 = gt_lab3.to(device, non_blocking=True).long()

            B = spc_feat.size(0)
            n_seen += B

            q = q_item.unsqueeze(0).expand(B, -1, -1).contiguous()
            pred_pairs = atlas_pairs.unsqueeze(0).expand(B, -1, -1).contiguous()
            pred_lab3 = atlas_lab3.unsqueeze(0).expand(B, -1, -1).contiguous() if atlas_lab3 is not None else None

            opt.zero_grad(set_to_none=True)

            with autocast_ctx(enabled=amp_runtime):
                pred_l1, pred_l2 = model(q, spc_feat)

                if label_dim == 2:
                    cd1 = torch.tensor(0.0, device=device) if w1 == 0.0 else compute_loss_labeldim2(
                        pred_l1, gt_xyz, pred_pairs, gt_pairs, args.cd_mode, args.label_cd_wx, args.label_cd_wy
                    )
                    cd2 = compute_loss_labeldim2(
                        pred_l2, gt_xyz, pred_pairs, gt_pairs, args.cd_mode, args.label_cd_wx, args.label_cd_wy
                    )
                    comps2 = None
                else:
                    if gt_lab3 is None:
                        raise RuntimeError("label_dim=3 but gt_lab3 is None (check dataset / gt_label3_array).")
                    if pred_lab3 is None:
                        raise RuntimeError("label_dim=3 but atlas_lab3 is None (check atlas_label3_array).")

                    cd1 = torch.tensor(0.0, device=device) if w1 == 0.0 else compute_loss_labeldim3_three_terms(
                        pred_l1, gt_xyz,
                        pred_pairs, gt_pairs,
                        pred_lab3, gt_lab3,
                        args.label_cd_wx, args.label_cd_wy,
                        args.cd3_w_plain, args.cd3_w_pair2, args.cd3_w_label3
                    )[0]

                    cd2, comps2 = compute_loss_labeldim3_three_terms(
                        pred_l2, gt_xyz,
                        pred_pairs, gt_pairs,
                        pred_lab3, gt_lab3,
                        args.label_cd_wx, args.label_cd_wy,
                        args.cd3_w_plain, args.cd3_w_pair2, args.cd3_w_label3
                    )

                cd_total = (w1 * cd1) + (w2 * cd2)

            # Laplacian + EXTRA losses in FP32
            with torch.cuda.amp.autocast(enabled=False):
                verts_f = pred_l2.float()
                faces_b = atlas_faces.unsqueeze(0).expand(B, -1, -1).contiguous()
                meshes_pred = Meshes(verts=verts_f, faces=faces_b)
                lap_l = mesh_laplacian_smoothing(meshes_pred, method=args.lap_method).float()

                # -------- NEW: per-chamber CD (dim=2 uses gt_pairs; dim=3 uses gt_lab3) --------
                chcd_sum_t = verts_f.new_zeros(())
                chcd_by_label_t: Dict[int, torch.Tensor] = {l: verts_f.new_zeros(()) for l in LABEL_IDS}

                if args.use_chamber_cd:
                    if label_dim == 2:
                        y_labels = gt_pairs
                    else:
                        if gt_lab3 is None:
                            raise RuntimeError("use_chamber_cd requires gt_lab3 when label_dim=3.")
                        y_labels = gt_lab3

                    chcd_sum_t, chcd_by_label_t = label_subset_cd_batch_mean(
                        pred_verts=verts_f,
                        y_tgt=gt_xyz.float(),
                        y_labels=y_labels,
                        pred_idx_by_label=pred_idx_by_label,
                        label_ids=LABEL_IDS,
                    )

                # -------- NEW: per-chamber Lap (ONLY dim=3) --------
                chlap_sum_t = verts_f.new_zeros(())
                chlap_by_label_t: Dict[int, torch.Tensor] = {l: verts_f.new_zeros(()) for l in LABEL_IDS}

                if args.use_chamber_lap and (faces_by_label_torch is not None):
                    for l in LABEL_IDS:
                        f_l = faces_by_label_torch.get(int(l), None)
                        if f_l is None or f_l.numel() == 0:
                            chlap_by_label_t[int(l)] = verts_f.new_zeros(())
                            continue
                        faces_l_b = f_l.unsqueeze(0).expand(B, -1, -1).contiguous()
                        meshes_l = Meshes(verts=verts_f, faces=faces_l_b)
                        lap_l_ch = mesh_laplacian_smoothing(meshes_l, method=args.lap_method).float()
                        chlap_by_label_t[int(l)] = lap_l_ch
                        chlap_sum_t = chlap_sum_t + lap_l_ch

                # final loss
                loss = (
                    float(args.cd_weight) * cd_total.float()
                    + float(lap_w) * lap_l
                    + float(args.chamber_cd_weight) * chcd_sum_t
                    + float(chamber_lap_weight) * chlap_sum_t
                )

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

            scaler.step(opt)
            scaler.update()

            # epoch sums
            sum_total += float(loss.item()) * B
            sum_cd1 += float(cd1.item()) * B
            sum_cd2 += float(cd2.item()) * B
            sum_lap += float(lap_l.item()) * B

            if comps2 is not None:
                l0, l2, l3 = comps2
                sum_cd2_plain += float(l0.item()) * B
                sum_cd2_pair2 += float(l2.item()) * B
                sum_cd2_lab3 += float(l3.item()) * B

            # extra sums
            sum_chcd += float(chcd_sum_t.item()) * B
            for l in LABEL_IDS:
                sum_chcd_by_label[int(l)] += float(chcd_by_label_t[int(l)].item()) * B

            sum_chlap += float(chlap_sum_t.item()) * B
            for l in LABEL_IDS:
                sum_chlap_by_label[int(l)] += float(chlap_by_label_t[int(l)].item()) * B

            # per-iteration prints
            if (it % args.print_every) == 0:
                lr_now = opt.param_groups[0]['lr']
                if label_dim == 2:
                    msg = (
                        f"Epoch {epoch:03d} Iter {it:04d}: "
                        f"loss={loss.item():.6f} cd1={cd1.item():.6f} cd2={cd2.item():.6f} "
                        f"lap={lap_l.item():.6f} "
                        f"chcd_sum={chcd_sum_t.item():.6f} "
                        f"lr={lr_now:.2e} amp={amp_runtime} bs={int(args.batch_size)} grad_clip={float(args.grad_clip):.6f}"
                    )
                    if args.use_chamber_cd:
                        for l in LABEL_IDS:
                            msg += f" chcd{l}={chcd_by_label_t[int(l)].item():.6f}"
                    print(msg)
                else:
                    l0, l2, l3 = comps2
                    msg = (
                        f"Epoch {epoch:03d} Iter {it:04d}: "
                        f"loss={loss.item():.6f} cd1={cd1.item():.6f} cd2={cd2.item():.6f} "
                        f"[plain={l0.item():.6f} pair2={l2.item():.6f} lab3={l3.item():.6f}] "
                        f"lap={lap_l.item():.6f} "
                        f"chcd_sum={chcd_sum_t.item():.6f} chlap_sum={chlap_sum_t.item():.6f} "
                        f"lr={lr_now:.2e} amp={amp_runtime} bs={int(args.batch_size)} grad_clip={float(args.grad_clip):.6f}"
                    )
                    if args.use_chamber_cd:
                        for l in LABEL_IDS:
                            msg += f" chcd{l}={chcd_by_label_t[int(l)].item():.6f}"
                    if args.use_chamber_lap:
                        for l in LABEL_IDS:
                            msg += f" chlap{l}={chlap_by_label_t[int(l)].item():.6f}"
                    print(msg)

        return (
            sum_total, sum_cd1, sum_cd2, sum_lap, n_seen,
            sum_cd2_plain, sum_cd2_pair2, sum_cd2_lab3,
            sum_chcd, sum_chcd_by_label,
            sum_chlap, sum_chlap_by_label,
        )

    MAX_EPOCH_RETRIES = int(args.max_epoch_retries)

    for epoch in range(start_epoch, args.epochs + 1):
        if args.ramp:
            ramp = min(1.0, epoch / max(1, int(0.3 * args.epochs)))
            lap_w = float(args.lap_weight) * ramp
            chamber_lap_weight = float(args.chamber_lap_weight) * ramp
        else:
            lap_w = float(args.lap_weight)
            chamber_lap_weight = float(args.chamber_lap_weight)

        epoch_start = {
            "model": copy.deepcopy(model.state_dict()),
            "opt": copy.deepcopy(opt.state_dict()),
            "sched": copy.deepcopy(sched.state_dict()),
            "scaler": copy.deepcopy(scaler.state_dict()),
            "rng": _get_rng_state(),
            "grad_clip": float(args.grad_clip),
            "batch_size": int(args.batch_size),
            "amp_runtime": bool(amp_runtime),
        }

        retry = 0
        while True:
            try:
                (
                    sum_total, sum_cd1, sum_cd2, sum_lap, n_seen,
                    sum_cd2_plain, sum_cd2_pair2, sum_cd2_lab3,
                    sum_chcd, sum_chcd_by_label,
                    sum_chlap, sum_chlap_by_label,
                ) = _run_one_epoch(epoch, lap_w)
                break
            except AssertionError as e:
                if not _looks_like_underflow_assertion(e):
                    raise
                retry += 1

                if retry > MAX_EPOCH_RETRIES:
                    old_bs = int(args.batch_size)
                    new_bs = max(1, old_bs // 2)
                    can_disable_amp = bool(amp_runtime)
                    can_shrink_bs = (new_bs < old_bs)

                    if can_disable_amp or can_shrink_bs:
                        print(
                            f"[FALLBACK] Epoch {epoch} exceeded max retries ({MAX_EPOCH_RETRIES}). "
                            f"Disable AMP={can_disable_amp}, bs {old_bs}->{new_bs}. "
                            f"Original error: {repr(e)}"
                        )
                        if can_disable_amp:
                            args.amp = False
                            autocast_ctx, scaler, amp_runtime = make_autocast_and_scaler(args.amp, device)
                        if can_shrink_bs:
                            args.batch_size = new_bs
                            loader = make_loader(int(args.batch_size))

                        model.load_state_dict(epoch_start["model"], strict=True)
                        opt.load_state_dict(epoch_start["opt"])
                        sched.load_state_dict(epoch_start["sched"])
                        scaler.load_state_dict(epoch_start["scaler"])
                        _set_rng_state(epoch_start["rng"])
                        args.grad_clip = float(epoch_start["grad_clip"])

                        opt.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                        retry = 0
                        continue

                    print(f"[FATAL] Epoch {epoch}: exceeded retries and fallback. Re-raising.")
                    raise

                print(
                    f"[RECOVER] Underflow-like AssertionError at epoch {epoch}. "
                    f"Retry {retry}/{MAX_EPOCH_RETRIES}. Restoring epoch-start and reducing grad_clip."
                )
                model.load_state_dict(epoch_start["model"], strict=True)
                opt.load_state_dict(epoch_start["opt"])
                sched.load_state_dict(epoch_start["sched"])
                scaler.load_state_dict(epoch_start["scaler"])
                _set_rng_state(epoch_start["rng"])

                args.grad_clip = float(epoch_start["grad_clip"]) * (0.90 ** retry)

                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

        sched.step()

        avg_loss = sum_total / max(1, n_seen)
        avg_cd1 = sum_cd1 / max(1, n_seen)
        avg_cd2 = sum_cd2 / max(1, n_seen)
        avg_lap = sum_lap / max(1, n_seen)
        lr_now = sched.get_last_lr()[0]

        # extras
        avg_chcd = sum_chcd / max(1, n_seen)
        avg_chlap = sum_chlap / max(1, n_seen)
        avg_chcd_by_label = {l: (sum_chcd_by_label[l] / max(1, n_seen)) for l in LABEL_IDS}
        avg_chlap_by_label = {l: (sum_chlap_by_label[l] / max(1, n_seen)) for l in LABEL_IDS}

        line = (
            f"Epoch {epoch:03d}: avg_loss={avg_loss:.6f} avg_cd1={avg_cd1:.6f} "
            f"avg_cd2={avg_cd2:.6f} avg_lap={avg_lap:.6f} lr={lr_now:.2e} "
            f"amp={amp_runtime} label_dim={label_dim} bs={int(args.batch_size)} grad_clip={float(args.grad_clip):.6f} "
            f"| chcd(avg)={avg_chcd:.6f} chlap(avg)={avg_chlap:.6f} "
            f"use_chcd={int(bool(args.use_chamber_cd))} use_chlap={int(bool(args.use_chamber_lap))}"
        )

        if label_dim == 3:
            avg_cd2_plain = sum_cd2_plain / max(1, n_seen)
            avg_cd2_pair2 = sum_cd2_pair2 / max(1, n_seen)
            avg_cd2_lab3 = sum_cd2_lab3 / max(1, n_seen)
            line += f" | cd2_plain={avg_cd2_plain:.6f} cd2_pair2={avg_cd2_pair2:.6f} cd2_lab3={avg_cd2_lab3:.6f}"

        print(line)
        with open(args.log_file, "a") as f:
            f.write(line + "\n")
            # per-label breakdown line (requested)
            if args.use_chamber_cd:
                f.write("  chamber_cd_by_label: " + " ".join([f"l{l}={avg_chcd_by_label[l]:.6f}" for l in LABEL_IDS]) + "\n")
            if args.use_chamber_lap:
                f.write("  chamber_lap_by_label: " + " ".join([f"l{l}={avg_chlap_by_label[l]:.6f}" for l in LABEL_IDS]) + "\n")

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "sched_state": sched.state_dict(),
            "scaler_state": scaler.state_dict(),
            "avg_loss": avg_loss,
            "avg_cd1": avg_cd1,
            "avg_cd2": avg_cd2,
            "avg_lap": avg_lap,
            "avg_chamber_cd": avg_chcd,
            "avg_chamber_lap": avg_chlap,
            "avg_chamber_cd_by_label": avg_chcd_by_label,
            "avg_chamber_lap_by_label": avg_chlap_by_label,
            "label_dim": label_dim,
            "spc_feature_dim": spc_feature_dim,
            "atlas_feature_dim": atlas_feature_dim,
            "d_points_spc": d_points_spc,
            "d_points_atlas": d_points_atlas,
            "cd_mode": (args.cd_mode if label_dim == 2 else "three_terms"),
            "cd3_w_plain": float(args.cd3_w_plain),
            "cd3_w_pair2": float(args.cd3_w_pair2),
            "cd3_w_label3": float(args.cd3_w_label3),
            "l1_loss_weight": w1,
            "l2_loss_weight": w2,
            "lap_weight": float(lap_w),
            "label_scale": float(args.label_scale),
            "label_cd_wx": float(args.label_cd_wx),
            "label_cd_wy": float(args.label_cd_wy),
            "use_chamber_cd": bool(args.use_chamber_cd),
            "chamber_cd_weight": float(args.chamber_cd_weight),
            "use_chamber_lap": bool(args.use_chamber_lap),
            "chamber_lap_weight": float(args.chamber_lap_weight),
            "label_ids": LABEL_IDS,
            "amp": bool(args.amp),
            "amp_runtime": bool(amp_runtime),
            "ode_force_fp32": bool(args.ode_force_fp32),
            "grad_clip": float(args.grad_clip),
            "batch_size": int(args.batch_size),
            "max_epoch_retries": int(MAX_EPOCH_RETRIES),
            "gt_pair_array": gt_pair_array,
            "atlas_pair_array": atlas_pair_array,
            "gt_label3_array": gt_label3_array,
            "atlas_label3_array": atlas_label3_array,
            "ca_type": args.ca_type,
            # ---- architecture (so scripts/infer.py can auto-resolve the model) ----
            "time": float(args.time),
            "tol": float(args.tol),
            "num_hidden": int(args.num_hidden),
            "latent_len": int(args.latent_len),
            "tanh_dynamics": bool(args.tanh_dynamics),
            "ca_nblocks": int(args.ca_nblocks),
            "ca_nneighbor": int(args.ca_nneighbor),
            "norm_type": str(args.norm_type),
            "cond_mode": str(args.cond_mode),
            "cond_norm": str(args.cond_norm),
            "ca_dec_fc_norm_type": str(args.ca_dec_fc_norm_type),
            "ca_dec_fp_norm_type": str(args.ca_dec_fp_norm_type),
            "affine_dynamics": bool(args.affine_dynamics),
        }

        tag = (
            f"lab{label_dim}"
            f"_q{atlas_feature_dim}"
            f"_w1-{w1:.2f}"
            f"_amp-{int(bool(args.amp))}"
            f"_bs-{int(args.batch_size)}"
            f"_ca-{args.ca_type}"
            f"_chcd-{int(bool(args.use_chamber_cd))}"
            f"_chlap-{int(bool(args.use_chamber_lap))}"
        )
        latest_path = os.path.join(args.ckpt_dir, f"{args.ckpt_prefix}_{tag}_latest.pt")
        torch.save(state, latest_path)

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"{args.ckpt_prefix}_{tag}_epoch{epoch:03d}.pt")
            torch.save(state, ckpt_path)


def parse_args():
    p = argparse.ArgumentParser()

    # ---------------- data ----------------
    p.add_argument("--sparse-dir", type=str, default="data/train/spc_ds")
    p.add_argument("--sparse-pcs-key", type=str, default="pcs")
    p.add_argument("--sparse-pairs-key", type=str, default="pairs")  # (N,2) ALWAYS

    p.add_argument("--label-dim", type=int, default=2, choices=[2, 3])
    p.add_argument("--no-label-feat", action="store_true",
                   help="Ablation: remove label channels from spc/atlas features (use BN6 => d_points=3).")

    # label_dim=2 paths/arrays
    p.add_argument("--gt-dir-2", type=str, default="data/train/gt_ds_5632_vtp")
    p.add_argument("--atlas-path-2", type=str, default="data/atlas.vtp")
    p.add_argument("--gt-pair-array-2", type=str, default="VertexBoundaryLabels")
    p.add_argument("--atlas-pair-array-2", type=str, default="VertexBoundaryLabels")
    # unused but kept
    p.add_argument("--gt-label3-array-2", type=str, default="VertexLabels3")
    p.add_argument("--atlas-label3-array-2", type=str, default="VertexLabels3")

    # label_dim=3 paths/arrays
    p.add_argument("--gt-dir-3", type=str, default="data/train/gt_ds_3lab_vtp")
    p.add_argument("--atlas-path-3", type=str, default="data/atlas_3lab.vtp")
    # pair arrays (for pair-CD)
    p.add_argument("--gt-pair-array-3", type=str, default="VertexBoundaryLabels")
    p.add_argument("--atlas-pair-array-3", type=str, default="VertexBoundaryLabels")
    # label3 arrays (for atlas BN9 feature + label3-CD + chamber lap)
    p.add_argument("--gt-label3-array-3", type=str, default="VertexLabels3")
    p.add_argument("--atlas-label3-array-3", type=str, default="VertexLabels3")

    # ---------------- output / resume ----------------
    p.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    p.add_argument("--ckpt-prefix", type=str, default="recon_model")
    p.add_argument("--resume", type=str, default=None)

    # ---------------- train loop ----------------
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-affine-dynamics", dest="affine_dynamics", action="store_false")
    p.set_defaults(affine_dynamics=True)

    # ---------------- optimizer ----------------
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=0.1)

    # ---------------- logging ----------------
    p.add_argument("--log-file", type=str, default="auto")  # AUTO -> ckpt_dir/train.txt
    p.add_argument("--print-every", type=int, default=2)
    p.add_argument("--save-every", type=int, default=50)

    # ---------------- precision ----------------
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)

    p.add_argument("--no-ode-force-fp32", dest="ode_force_fp32", action="store_false")
    p.set_defaults(ode_force_fp32=True)

    p.add_argument("--max-epoch-retries", type=int, default=5)

    # ---------------- model hyperparams ----------------
    p.add_argument("--time", type=float, default=0.2)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--num-hidden", type=int, default=512)
    p.add_argument("--latent-len", type=int, default=32)
    p.add_argument("--tanh-dynamics", action="store_true")
    p.set_defaults(tanh_dynamics=True)

    p.add_argument("--ca-type", type=str, default="double", choices=["single", "double"])

    p.add_argument("--ca-nblocks", type=int, default=4)
    p.add_argument("--ca-nneighbor", type=int, default=16)

    p.add_argument("--norm-type", type=str, default="in", choices=["bn", "in"])
    p.add_argument("--cond-mode", type=str, default="concat", choices=["concat", "mul", "add", "code", "global"])
    p.add_argument("--cond-norm", type=str, default="none", choices=["none", "layernorm", "l2"])

    p.add_argument("--ca-dec-fc-norm-type", type=str, default="bn", choices=["auto", "bn", "in"])
    p.add_argument("--ca-dec-fp-norm-type", type=str, default="in", choices=["auto", "bn", "in"])

    # ---------------- loss weights ----------------
    p.add_argument("--cd-weight", type=float, default=1.0)
    p.add_argument("--l1-loss-weight", type=float, default=0.3)
    p.add_argument("--lap-weight", type=float, default=1.0)
    p.add_argument("--ramp", action="store_true")
    p.add_argument("--lap-method", type=str, default="uniform", choices=["uniform", "cot", "cotcurv"])

    # label_dim=2 only (ignored if label_dim=3)
    p.add_argument("--cd-mode", type=str, default="both", choices=["plain", "label", "both"])

    p.add_argument("--label-scale", type=float, default=10.0)
    p.add_argument("--label-cd-wx", type=float, default=1.0)
    p.add_argument("--label-cd-wy", type=float, default=1.2)

    # label_dim=3 only: three-term weights
    p.add_argument("--cd3-w-plain", type=float, default=1.0)
    p.add_argument("--cd3-w-pair2", type=float, default=1.0)
    p.add_argument("--cd3-w-label3", type=float, default=1.0)

    # ---------------- NEW: optional per-chamber losses ----------------
    p.add_argument("--use-chamber-cd", action="store_true",
                   help="Add per-chamber subset CD (label_dim=2 uses VertexBoundaryLabels; label_dim=3 uses VertexLabels3).")
    p.add_argument("--chamber-cd-weight", type=float, default=1.0,
                   help="Weight multiplying the SUM of per-chamber CDs.")

    p.add_argument("--use-chamber-lap", action="store_true",
                   help="Add per-chamber Laplacian regularization (ONLY label_dim=3; auto-disabled for label_dim=2).")
    p.add_argument("--chamber-lap-weight", type=float, default=1.0,
                   help="Weight multiplying the SUM of per-chamber Laplacians (dim=3 only).")

    return p.parse_args()


if __name__ == "__main__":
    from torch import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    args = parse_args()
    if not args.amp:
        args.ode_force_fp32 = False
    train(args)
