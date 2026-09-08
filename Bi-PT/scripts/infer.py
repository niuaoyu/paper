#!/usr/bin/env python3
"""Bi-PT inference: deform the atlas toward each sparse point cloud and write meshes.

Batched inference (+ optional Chamfer eval) for ``DeformBlockConcat`` /
``DeformBlockConcatSingleCA``. For each input SPC (``.npz``) it runs the model,
**denormalizes** the prediction back to world coordinates using that sample's own
normalization stats, and writes the predicted mesh as ``<case>_pred_l1.vtp`` and
``<case>_pred_l2.vtp`` (atlas faces + atlas label arrays). Those files are the
input to ``scripts/evaluate.py``.

Architecture is auto-loaded from the checkpoint when present (``time``,
``num_hidden``, ``latent_len``, ``ca_nblocks``, ``ca_type``, ``label_dim``,
``d_points_*``, ``label_scale``, ``affine_dynamics``, ...); override via CLI, and
use ``--strict-arch`` to hard-error on a stored/requested mismatch.

Inference is pure FP32 (no AMP; TF32 disabled on CUDA) for stricter numerics.

Shared IO / feature building is imported from :mod:`bipt.data.dataset` so it stays
in lock-step with training; only the inference-specific pieces (denormalization
and mesh writing) live here.

Example:
  python scripts/infer.py \
    --ckpt checkpoints/recon_model_lab2_q8_w1-0.30_amp-1_bs-8_ca-double_latest.pt \
    --spc-dir data/test/spc_ds --gt-dir data/test/gt_ds_5632_vtp \
    --atlas-path-2 data/atlas.vtp --eval-cd both --out-dir outputs/infer \
    --save-metrics outputs/infer/cd_metrics.csv
"""

import os
import sys
import glob
import csv
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch

import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

# Allow running as `python scripts/infer.py` from the repo root without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bipt.models import DeformBlockConcat, DeformBlockConcatSingleCA
from bipt.losses import chamfer_distance as chamfer_distance_fn
from bipt.data.dataset import (
    pc_normalize_xyz,
    _canonicalize_rows,
    build_spc_feat_bn6,
    build_atlas_feat_bn6,
    build_spc_feat_bn8,
    build_atlas_feat,
    load_vtp_points_and_labels,
    load_vtp_faces,
    load_sparse_npz_points_and_pairs,
)


# ----------------------------- device & strict FP32 -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# =========================
# Denormalization (invert bipt.data.dataset.pc_normalize_xyz)
# =========================
def compute_norm_params_from_sparse(sparse_xyz: np.ndarray) -> Dict[str, np.ndarray]:
    """Save the exact params needed to invert pc_normalize_xyz() for this sparse sample."""
    sparse_xyz = sparse_xyz.astype(np.float32)
    centroid = sparse_xyz.mean(axis=0, keepdims=True)
    centered = sparse_xyz - centroid
    mins = centered.min(axis=0)                 # (3,)
    maxs = centered.max(axis=0)                 # (3,)
    spans = np.maximum(maxs - mins, 1e-8)       # (3,)
    return {"centroid": centroid, "mins": mins, "spans": spans}


def denormalize_with_params(xyz_norm: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """Invert scaling from [-0.85, 0.85] back to original coords using saved mins/spans/centroid."""
    xyz_norm = np.asarray(xyz_norm, dtype=np.float32)
    centroid, mins, spans = params["centroid"], params["mins"], params["spans"]

    centered = np.empty_like(xyz_norm, dtype=np.float32)
    centered[:, 0] = (xyz_norm[:, 0] + 0.85) * spans[0] / 1.7 + mins[0]
    centered[:, 1] = (xyz_norm[:, 1] + 0.85) * spans[1] / 1.7 + mins[1]
    centered[:, 2] = (xyz_norm[:, 2] + 0.85) * spans[2] / 1.7 + mins[2]
    return centered + centroid


# =========================
# Mesh writer
# =========================
def write_mesh_vtp(
    coords_xyz: np.ndarray,
    faces: np.ndarray,
    out_vtp: str,
    pair2: Optional[np.ndarray],
    label3: Optional[np.ndarray],
    pair_array_name: str = "VertexBoundaryLabels",
    label3_array_name: str = "VertexLabels3",
):
    coords_xyz = np.asarray(coords_xyz, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if coords_xyz.ndim != 2 or coords_xyz.shape[1] != 3:
        raise ValueError(f"coords must be (N,3), got {coords_xyz.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {faces.shape}")

    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(coords_xyz, deep=True))

    nF = faces.shape[0]
    cells = np.hstack([np.full((nF, 1), 3, dtype=np.int64), faces]).reshape(-1)
    vtk_ids = numpy_to_vtkIdTypeArray(cells, deep=True)

    ca = vtk.vtkCellArray()
    ca.SetCells(nF, vtk_ids)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(ca)

    N = coords_xyz.shape[0]

    if pair2 is not None:
        pair2 = np.asarray(pair2)
        if pair2.shape[0] != N or pair2.ndim != 2 or pair2.shape[1] < 2:
            raise ValueError(f"pair2 must be (N,2), got {pair2.shape}")
        pair2 = _canonicalize_rows(pair2)[:, :2].astype(np.int64)
        arr = numpy_to_vtk(pair2, deep=True)
        arr.SetName(pair_array_name)
        poly.GetPointData().AddArray(arr)

    if label3 is not None:
        label3 = np.asarray(label3)
        if label3.shape[0] != N or label3.ndim != 2 or label3.shape[1] < 3:
            raise ValueError(f"label3 must be (N,3), got {label3.shape}")
        label3 = _canonicalize_rows(label3)[:, :3].astype(np.int64)
        arr3 = numpy_to_vtk(label3, deep=True)
        arr3.SetName(label3_array_name)
        poly.GetPointData().AddArray(arr3)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_vtp)
    writer.SetInputData(poly)
    writer.Write()


# =========================
# Chamfer eval wrappers (use bipt.losses.chamfer_distance)
# =========================
def cd_plain(pred_bnx3: torch.Tensor, gt_bmx3: torch.Tensor) -> float:
    cd, _ = chamfer_distance_fn(
        pred_bnx3, gt_bmx3,
        batch_reduction="mean", point_reduction="mean", norm=2,
        single_directional=False, x_labels=None, y_labels=None,
    )
    return float(cd.item())


def cd_labeled(
    pred_bnx3: torch.Tensor,
    gt_bmx3: torch.Tensor,
    x_labels: torch.Tensor,
    y_labels: torch.Tensor,
    wx: float,
    wy: float,
) -> float:
    cd, _ = chamfer_distance_fn(
        pred_bnx3, gt_bmx3,
        batch_reduction="mean", point_reduction="mean", norm=2,
        single_directional=False,
        x_labels=x_labels, y_labels=y_labels,
        weight_x=float(wx), weight_y=float(wy),
    )
    return float(cd.item())


# =========================
# ckpt auto-resolvers
# =========================
def _resolve_auto(val, ckpt: dict, key: str, default):
    ck = ckpt.get(key, None)
    if val == "auto":
        return (ck if ck is not None else default), ck
    return val, ck


def _resolve_bool_auto(val: str, ckpt: dict, key: str, default: bool):
    ck = ckpt.get(key, None)
    if val == "auto":
        return (bool(ck) if ck is not None else bool(default)), ck
    s = str(val).lower()
    if s in ("true", "1", "yes", "y"):
        return True, ck
    if s in ("false", "0", "no", "n"):
        return False, ck
    raise ValueError(f"Invalid {key}='{val}'. Use auto/true/false.")


def _auto_to_none(s: str) -> Optional[str]:
    if s is None:
        return None
    s = str(s).lower()
    return None if s == "auto" else s


def _strict_check(args, ckval, reqval, key: str):
    if not args.strict_arch:
        return
    if ckval is None:
        return
    if isinstance(reqval, bool):
        if bool(ckval) != bool(reqval):
            raise RuntimeError(f"Checkpoint {key}={ckval} mismatches requested {key}={reqval}")
    else:
        if str(ckval) != str(reqval):
            raise RuntimeError(f"Checkpoint {key}={ckval} mismatches requested {key}={reqval}")


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="Bi-PT inference (DeformBlockConcat / SingleCA) with auto hyperparams from ckpt."
    )

    # ---- checkpoint ----
    p.add_argument("--ckpt", type=str, required=True)

    # ---- input sparse ----
    p.add_argument("--spc", type=str, default=None, help="single .npz")
    p.add_argument("--spc-dir", type=str, default=None, help="directory of .npz")
    p.add_argument("--spc-pcs-key", type=str, default="pcs")
    p.add_argument("--spc-pairs-key", type=str, default="pairs")

    # ---- label mode ----
    p.add_argument("--label-dim", type=str, default="auto", choices=["auto", "2", "3"])
    p.add_argument("--no-label-feat", type=str, default="auto",
                   choices=["auto", "true", "false"],
                   help="auto uses ckpt feature dims if present; true=>BN6; false=>label-feat BN8/BN9.")

    # ---- atlas paths/arrays ----
    p.add_argument("--atlas-path-2", type=str, default="data/atlas.vtp")
    p.add_argument("--atlas-path-3", type=str, default="data/atlas_3lab.vtp")
    p.add_argument("--atlas-pair-array", type=str, default="VertexBoundaryLabels")
    p.add_argument("--atlas-label3-array", type=str, default="VertexLabels3")

    # ---- GT optional eval ----
    p.add_argument("--gt", type=str, default=None)
    p.add_argument("--gt-dir", type=str, default=None)
    p.add_argument("--gt-pair-array", type=str, default="VertexBoundaryLabels")
    p.add_argument("--gt-label3-array", type=str, default="VertexLabels3")

    # ---- output ----
    p.add_argument("--out-dir", type=str, default="outputs/infer")
    p.add_argument("--which-output", type=str, default="both", choices=["l1", "l2", "both"])
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)

    # ---- save labels on pred mesh ----
    p.add_argument("--write-pair2", type=str, default="auto", choices=["auto", "true", "false"])
    p.add_argument("--write-label3", type=str, default="auto", choices=["auto", "true", "false"])

    # ---- evaluation ----
    p.add_argument("--eval-cd", type=str, default="both",
                   choices=["none", "plain", "label", "pair2", "both", "three"])
    p.add_argument("--label-cd-wx", type=float, default=1.0)
    p.add_argument("--label-cd-wy", type=float, default=1.0)
    p.add_argument("--save-metrics", type=str, default=None)

    # ---- label_scale ----
    p.add_argument("--label-scale", type=str, default="auto", help="auto uses ckpt['label_scale'] else 10.0")

    # ---- arch overrides (auto from ckpt) ----
    p.add_argument("--ca-type", type=str, default="auto", choices=["auto", "single", "double"])
    p.add_argument("--time", type=str, default="auto")
    p.add_argument("--tol", type=str, default="auto")
    p.add_argument("--num-hidden", type=str, default="auto")
    p.add_argument("--latent-len", type=str, default="auto")
    p.add_argument("--tanh-dynamics", type=str, default="auto", choices=["auto", "true", "false"])
    p.add_argument("--ca-nblocks", type=str, default="auto")
    p.add_argument("--ca-nneighbor", type=str, default="auto")
    p.add_argument("--norm-type", type=str, default="auto", choices=["auto", "bn", "in"])
    p.add_argument("--cond-mode", type=str, default="auto", choices=["auto", "concat", "mul", "add", "code", "global"])
    p.add_argument("--cond-norm", type=str, default="auto", choices=["auto", "none", "layernorm", "l2"])
    p.add_argument("--ca-dec-fc-norm-type", type=str, default="auto", choices=["auto", "bn", "in"])
    p.add_argument("--ca-dec-fp-norm-type", type=str, default="auto", choices=["auto", "bn", "in"])
    p.add_argument("--ode-force-fp32", type=str, default="auto", choices=["auto", "true", "false"])

    # ---- affine dynamics (MATCH TRAINING FLAG NAME) ----
    p.add_argument("--no-affine-dynamics", dest="affine_dynamics", action="store_false")
    p.set_defaults(affine_dynamics=True)

    # ---- strict ----
    p.add_argument("--strict-arch", action="store_true")

    return p.parse_args()


def _match_gt_path(spc_npz_path: str, gt_dir: str) -> str:
    base = os.path.splitext(os.path.basename(spc_npz_path))[0]
    return os.path.join(gt_dir, f"{base}.vtp")


# =========================
# Main
# =========================
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_metrics is not None:
        os.makedirs(os.path.dirname(args.save_metrics) or ".", exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state", ckpt)

    # ----- label_dim (auto) -----
    ck_label_dim = ckpt.get("label_dim", None)
    if args.label_dim == "auto":
        label_dim = int(ck_label_dim) if ck_label_dim is not None else 2
    else:
        label_dim = int(args.label_dim)
    if label_dim not in (2, 3):
        raise ValueError("label_dim must be 2 or 3")
    _strict_check(args, ck_label_dim, label_dim, "label_dim")

    # ----- label_scale (auto) -----
    label_scale_val, ck_ls = _resolve_auto(args.label_scale, ckpt, "label_scale", 10.0)
    label_scale = float(label_scale_val)
    _strict_check(args, ck_ls, label_scale, "label_scale")

    # ----- no_label_feat (auto via ckpt feature dims if present) -----
    ck_spc_feat_dim = ckpt.get("spc_feature_dim", None)
    if args.no_label_feat == "auto":
        no_label_feat = (int(ck_spc_feat_dim) == 6) if ck_spc_feat_dim is not None else False
    else:
        no_label_feat = (args.no_label_feat.lower() in ("true", "1", "yes", "y"))

    # ----- derive feature dims / d_points (prefer ckpt if present) -----
    if "d_points_spc" in ckpt:
        d_points_spc = int(ckpt["d_points_spc"])
    else:
        d_points_spc = 3 if no_label_feat else 5

    if "d_points_atlas" in ckpt:
        d_points_atlas = int(ckpt["d_points_atlas"])
    else:
        if no_label_feat:
            d_points_atlas = 3
        else:
            d_points_atlas = 5 if label_dim == 2 else 6

    # ----- choose ca_type / model class (auto) -----
    ca_type_val, ck_ca = _resolve_auto(args.ca_type, ckpt, "ca_type", "double")
    ca_type = str(ca_type_val)
    _strict_check(args, ck_ca, ca_type, "ca_type")
    ModelCls = DeformBlockConcat if ca_type == "double" else DeformBlockConcatSingleCA

    # ----- arch params (auto) -----
    time_val, ck_time = _resolve_auto(args.time, ckpt, "time", 0.2)
    tol_val, ck_tol = _resolve_auto(args.tol, ckpt, "tol", 1e-5)
    nh_val, ck_nh = _resolve_auto(args.num_hidden, ckpt, "num_hidden", 512)
    ll_val, ck_ll = _resolve_auto(args.latent_len, ckpt, "latent_len", 32)
    tanh_val, ck_tanh = _resolve_bool_auto(args.tanh_dynamics, ckpt, "tanh_dynamics", True)
    nb_val, ck_nb = _resolve_auto(args.ca_nblocks, ckpt, "ca_nblocks", 4)
    nn_val, ck_nn = _resolve_auto(args.ca_nneighbor, ckpt, "ca_nneighbor", 16)

    norm_type_val, ck_norm = _resolve_auto(args.norm_type, ckpt, "norm_type", "in")
    cond_mode_val, ck_cm = _resolve_auto(args.cond_mode, ckpt, "cond_mode", "concat")
    cond_norm_val, ck_cn = _resolve_auto(args.cond_norm, ckpt, "cond_norm", "none")

    fc_val, ck_fc = _resolve_auto(args.ca_dec_fc_norm_type, ckpt, "ca_dec_fc_norm_type", "bn")
    fp_val, ck_fp = _resolve_auto(args.ca_dec_fp_norm_type, ckpt, "ca_dec_fp_norm_type", "in")
    fc_norm = _auto_to_none(str(fc_val))
    fp_norm = _auto_to_none(str(fp_val))

    ode_val, ck_ode = _resolve_bool_auto(args.ode_force_fp32, ckpt, "ode_force_fp32", True)

    # ----- affine_dynamics (auto from ckpt unless user passes --no-affine-dynamics) -----
    ck_aff = ckpt.get("affine_dynamics", None)
    cli_overrode_affine = ("--no-affine-dynamics" in sys.argv)
    if cli_overrode_affine:
        affine_dynamics = bool(args.affine_dynamics)  # False
    else:
        affine_dynamics = (bool(ck_aff) if ck_aff is not None else True)  # training default True
    _strict_check(args, ck_aff, bool(affine_dynamics), "affine_dynamics")

    # strict checks if requested
    _strict_check(args, ck_time, float(time_val), "time")
    _strict_check(args, ck_tol, float(tol_val), "tol")
    _strict_check(args, ck_nh, int(nh_val), "num_hidden")
    _strict_check(args, ck_ll, int(ll_val), "latent_len")
    _strict_check(args, ck_tanh, bool(tanh_val), "tanh_dynamics")
    _strict_check(args, ck_nb, int(nb_val), "ca_nblocks")
    _strict_check(args, ck_nn, int(nn_val), "ca_nneighbor")
    _strict_check(args, ck_norm, str(norm_type_val), "norm_type")
    _strict_check(args, ck_cm, str(cond_mode_val), "cond_mode")
    _strict_check(args, ck_cn, str(cond_norm_val), "cond_norm")
    _strict_check(args, ck_fc, str(fc_val), "ca_dec_fc_norm_type")
    _strict_check(args, ck_fp, str(fp_val), "ca_dec_fp_norm_type")
    _strict_check(args, ck_ode, bool(ode_val), "ode_force_fp32")

    # ----- atlas selection -----
    atlas_path = args.atlas_path_2 if label_dim == 2 else args.atlas_path_3
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Atlas not found: {atlas_path}")

    # load atlas: faces + pair2 always + label3 if needed
    atlas_xyz, atlas_pair2 = load_vtp_points_and_labels(atlas_path, args.atlas_pair_array, min_dim=2)
    atlas_faces = load_vtp_faces(atlas_path)

    atlas_label3 = None
    if label_dim == 3:
        _, atlas_label3 = load_vtp_points_and_labels(atlas_path, args.atlas_label3_array, min_dim=3)

    # build atlas q_item feature
    if no_label_feat:
        q_item_np = build_atlas_feat_bn6(atlas_xyz)  # (Na,6)
    else:
        if label_dim == 2:
            q_item_np = build_atlas_feat(atlas_xyz, atlas_pair2, feat_label_dim=2, label_scale=label_scale)  # (Na,8)
        else:
            if atlas_label3 is None:
                raise RuntimeError("label_dim=3 requires atlas label3 array.")
            q_item_np = build_atlas_feat(atlas_xyz, atlas_label3, feat_label_dim=3, label_scale=label_scale)  # (Na,9)

    q_item = torch.from_numpy(q_item_np).float().to(device)
    atlas_pair2_t = torch.from_numpy(atlas_pair2).long().to(device)  # (Na,2)
    atlas_label3_t = torch.from_numpy(atlas_label3).long().to(device) if atlas_label3 is not None else None
    faces_np = np.asarray(atlas_faces, dtype=np.int64)

    # ----- model -----
    model = ModelCls(
        time=float(time_val),
        num_hidden=int(nh_val),
        latent_len=int(ll_val),
        tol=float(tol_val),
        tanh_dynamics=bool(tanh_val),
        d_points_spc=int(d_points_spc),
        d_points_atlas=int(d_points_atlas),
        ca_nblocks=int(nb_val),
        ca_nneighbor=int(nn_val),
        norm_type=str(norm_type_val),
        cond_mode=str(cond_mode_val),
        cond_norm=str(cond_norm_val),
        ca_dec_fc_norm_type=fc_norm,
        ca_dec_fp_norm_type=fp_norm,
        ode_force_fp32=bool(ode_val),
        affine_dynamics=bool(affine_dynamics),
    ).to(device).float()

    model.load_state_dict(state, strict=True)
    model.eval()

    # ----- decide writing label arrays -----
    if args.write_pair2 == "auto":
        write_pair2 = True
    else:
        write_pair2, _ = _resolve_bool_auto(args.write_pair2, ckpt, "write_pair2", True)

    if args.write_label3 == "auto":
        write_label3 = (label_dim == 3)
    else:
        write_label3, _ = _resolve_bool_auto(args.write_label3, ckpt, "write_label3", (label_dim == 3))

    print(
        f"[Config]\n"
        f"  device={device.type} (PURE FP32)\n"
        f"  ckpt={args.ckpt}\n"
        f"  ca_type={ca_type}  label_dim={label_dim}  no_label_feat={no_label_feat}\n"
        f"  d_points_spc={d_points_spc}  d_points_atlas={d_points_atlas}\n"
        f"  time={float(time_val)} tol={float(tol_val)} num_hidden={int(nh_val)} latent_len={int(ll_val)}\n"
        f"  tanh_dynamics={bool(tanh_val)} ca_nblocks={int(nb_val)} ca_nneighbor={int(nn_val)}\n"
        f"  norm_type={norm_type_val} cond_mode={cond_mode_val} cond_norm={cond_norm_val}\n"
        f"  ca_dec_fc_norm_type={fc_norm} ca_dec_fp_norm_type={fp_norm}\n"
        f"  ode_force_fp32={bool(ode_val)} affine_dynamics={bool(affine_dynamics)}\n"
        f"  label_scale={label_scale}\n"
        f"  atlas={atlas_path} pair_array={args.atlas_pair_array} label3_array={args.atlas_label3_array}\n"
        f"  write_pair2={write_pair2} write_label3={write_label3}\n"
        f"  which_output={args.which_output} eval_cd={args.eval_cd} batch_size={args.batch_size}\n"
    )

    # ----- sparse inputs -----
    if args.spc is not None:
        spc_files = [args.spc]
    else:
        if args.spc_dir is None:
            raise RuntimeError("Provide either --spc or --spc-dir.")
        spc_files = sorted(glob.glob(os.path.join(args.spc_dir, "*.npz")))
    if not spc_files:
        raise RuntimeError("No sparse .npz files found.")

    # ----- GT optional (single or dir) -----
    single_gt = args.gt if (args.gt is not None and os.path.exists(args.gt)) else None
    need_gt = (args.eval_cd != "none")
    need_pair2_eval = (args.eval_cd in ("pair2", "both", "three")) or (label_dim == 2 and args.eval_cd in ("label", "both"))
    need_label3_eval = (label_dim == 3 and args.eval_cd in ("label", "three"))

    rows: List[Dict] = []

    # accumulators for per-case print + dataset mean
    plain_l1_vals: List[float] = []
    plain_l2_vals: List[float] = []
    pair2_l1_vals: List[float] = []
    pair2_l2_vals: List[float] = []
    label3_l1_vals: List[float] = []
    label3_l2_vals: List[float] = []
    n_missing_gt = 0
    n_eval = 0

    batch_size = max(1, int(args.batch_size))
    n_total = len(spc_files)

    for start in range(0, n_total, batch_size):
        chunk = spc_files[start:start + batch_size]

        items = []
        for spc_path in chunk:
            base = os.path.splitext(os.path.basename(spc_path))[0]
            sp_xyz, sp_pairs = load_sparse_npz_points_and_pairs(spc_path, args.spc_pcs_key, args.spc_pairs_key)
            sp_params = compute_norm_params_from_sparse(sp_xyz)

            if no_label_feat:
                x_feat = build_spc_feat_bn6(sp_xyz)  # (Ns,6)
            else:
                x_feat = build_spc_feat_bn8(sp_xyz, sp_pairs, label_scale=label_scale)  # (Ns,8)

            items.append((spc_path, base, sp_xyz, sp_pairs, x_feat.astype(np.float32), sp_params))

        # group by Ns to avoid padding
        groups: Dict[int, List] = {}
        for it in items:
            Ns = it[4].shape[0]
            groups.setdefault(Ns, []).append(it)

        for Ns, gitems in groups.items():
            x_batch = np.stack([gi[4] for gi in gitems], axis=0)  # (B,Ns,C)
            x = torch.from_numpy(x_batch).float().to(device, non_blocking=True)

            B = x.shape[0]
            q = q_item.unsqueeze(0).expand(B, -1, -1).contiguous()

            with torch.inference_mode():
                pred_l1, pred_l2 = model(q, x)

            pred_l1 = pred_l1.float()
            pred_l2 = pred_l2.float()

            for b in range(B):
                spc_path, base, sp_xyz, sp_pairs, _, sp_params = gitems[b]
                suffix = f"_{args.tag}" if args.tag else ""

                out_l1 = out_l2 = None

                def _save_one(pred_norm_t: torch.Tensor, stage: str) -> str:
                    pred_norm = pred_norm_t.detach().cpu().numpy().astype(np.float32)  # (Na,3)
                    pred_den = denormalize_with_params(pred_norm, sp_params)
                    out_vtp = os.path.join(args.out_dir, f"{base}_pred_{stage}{suffix}.vtp")

                    pair2 = atlas_pair2 if write_pair2 else None
                    lab3 = atlas_label3 if (write_label3 and atlas_label3 is not None) else None

                    write_mesh_vtp(
                        coords_xyz=pred_den,
                        faces=faces_np,
                        out_vtp=out_vtp,
                        pair2=pair2,
                        label3=lab3,
                        pair_array_name=args.atlas_pair_array,
                        label3_array_name=args.atlas_label3_array,
                    )
                    return out_vtp

                if args.which_output in ("l1", "both"):
                    out_l1 = _save_one(pred_l1[b], "l1")
                if args.which_output in ("l2", "both"):
                    out_l2 = _save_one(pred_l2[b], "l2")

                row = {
                    "case": base,
                    "spc_path": spc_path,
                    "out_vtp_l1": out_l1,
                    "out_vtp_l2": out_l2,
                    "cd_plain_l1": "",
                    "cd_plain_l2": "",
                    "cd_pair2_l1": "",
                    "cd_pair2_l2": "",
                    "cd_label3_l1": "",
                    "cd_label3_l2": "",
                }

                # -------- optional eval --------
                if not need_gt:
                    print(f"[INFER] {base} | no-eval (eval_cd=none)")
                    rows.append(row)
                    continue

                gt_path = None
                if single_gt is not None:
                    gt_path = single_gt
                elif args.gt_dir is not None:
                    gt_path = _match_gt_path(spc_path, args.gt_dir)

                if gt_path is None or (not os.path.exists(gt_path)):
                    n_missing_gt += 1
                    print(f"[INFER] {base} | missing GT")
                    rows.append(row)
                    continue

                gt_xyz, _ = load_vtp_points_and_labels(gt_path, args.gt_pair_array, min_dim=2)
                gt_norm = pc_normalize_xyz(sp_xyz, gt_xyz).astype(np.float32)
                gt_tensor = torch.from_numpy(gt_norm[None, ...]).float().to(device)

                p1 = pred_l1[b:b+1]
                p2 = pred_l2[b:b+1]

                n_eval += 1
                plain_l1 = plain_l2 = None
                pair2_l1 = pair2_l2 = None
                label3_l1 = label3_l2 = None

                do_plain = args.eval_cd in ("plain", "both", "three", "label", "pair2")
                if do_plain:
                    cd1 = cd_plain(p1, gt_tensor)
                    cd2 = cd_plain(p2, gt_tensor)
                    row["cd_plain_l1"] = f"{cd1:.6f}"
                    row["cd_plain_l2"] = f"{cd2:.6f}"
                    plain_l1, plain_l2 = cd1, cd2
                    plain_l1_vals.append(cd1)
                    plain_l2_vals.append(cd2)

                if need_pair2_eval:
                    _, gt_pair2 = load_vtp_points_and_labels(gt_path, args.gt_pair_array, min_dim=2)
                    x_labels = atlas_pair2_t.unsqueeze(0)  # (1,Na,2)
                    y_labels = torch.from_numpy(gt_pair2[None, ...]).long().to(device)  # (1,Ngt,2)

                    cd1 = cd_labeled(p1, gt_tensor, x_labels, y_labels, wx=args.label_cd_wx, wy=args.label_cd_wy)
                    cd2 = cd_labeled(p2, gt_tensor, x_labels, y_labels, wx=args.label_cd_wx, wy=args.label_cd_wy)
                    row["cd_pair2_l1"] = f"{cd1:.6f}"
                    row["cd_pair2_l2"] = f"{cd2:.6f}"
                    pair2_l1, pair2_l2 = cd1, cd2
                    pair2_l1_vals.append(cd1)
                    pair2_l2_vals.append(cd2)

                if need_label3_eval:
                    if atlas_label3_t is None:
                        raise RuntimeError("Requested label3 eval but atlas_label3 is missing.")
                    _, gt_lab3 = load_vtp_points_and_labels(gt_path, args.gt_label3_array, min_dim=3)
                    x_labels = atlas_label3_t.unsqueeze(0)  # (1,Na,3)
                    y_labels = torch.from_numpy(gt_lab3[None, ...]).long().to(device)  # (1,Ngt,3)

                    cd1 = cd_labeled(p1, gt_tensor, x_labels, y_labels, wx=args.label_cd_wx, wy=args.label_cd_wy)
                    cd2 = cd_labeled(p2, gt_tensor, x_labels, y_labels, wx=args.label_cd_wx, wy=args.label_cd_wy)
                    row["cd_label3_l1"] = f"{cd1:.6f}"
                    row["cd_label3_l2"] = f"{cd2:.6f}"
                    label3_l1, label3_l2 = cd1, cd2
                    label3_l1_vals.append(cd1)
                    label3_l2_vals.append(cd2)

                parts = [f"[INFER] {base}"]
                if plain_l1 is not None:
                    parts.append(f"plain(l1,l2)={plain_l1:.6f},{plain_l2:.6f}")
                if pair2_l1 is not None:
                    parts.append(f"pair2(l1,l2)={pair2_l1:.6f},{pair2_l2:.6f}")
                if label3_l1 is not None:
                    parts.append(f"label3(l1,l2)={label3_l1:.6f},{label3_l2:.6f}")
                print(" | ".join(parts))

                rows.append(row)

    # ----- save metrics -----
    if args.save_metrics is not None:
        with open(args.save_metrics, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "case", "spc_path", "out_vtp_l1", "out_vtp_l2",
                    "cd_plain_l1", "cd_plain_l2",
                    "cd_pair2_l1", "cd_pair2_l2",
                    "cd_label3_l1", "cd_label3_l2",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[Saved metrics] {args.save_metrics}")

    # summary means over dataset
    print("\n================ Summary ================")
    print(f"evaluated_with_gt={n_eval}  missing_gt={n_missing_gt}  total_cases={len(rows)}")

    if plain_l1_vals:
        print(f"[MEAN] plain_cd_l1 = {float(np.mean(plain_l1_vals)):.6f} (n={len(plain_l1_vals)})")
    if plain_l2_vals:
        print(f"[MEAN] plain_cd_l2 = {float(np.mean(plain_l2_vals)):.6f} (n={len(plain_l2_vals)})")
    if pair2_l1_vals:
        print(f"[MEAN] pair2_cd_l1 = {float(np.mean(pair2_l1_vals)):.6f} (n={len(pair2_l1_vals)})")
    if pair2_l2_vals:
        print(f"[MEAN] pair2_cd_l2 = {float(np.mean(pair2_l2_vals)):.6f} (n={len(pair2_l2_vals)})")
    if label3_l1_vals:
        print(f"[MEAN] label3_cd_l1 = {float(np.mean(label3_l1_vals)):.6f} (n={len(label3_l1_vals)})")
    if label3_l2_vals:
        print(f"[MEAN] label3_cd_l2 = {float(np.mean(label3_l2_vals)):.6f} (n={len(label3_l2_vals)})")

    if (not plain_l1_vals) and (not plain_l2_vals) and (not pair2_l2_vals) and (not label3_l2_vals):
        print("No CD values computed (check GT paths or eval_cd).")
    print("========================================")

    print(f"[Done] saved {len(rows)} cases to {args.out_dir}")


if __name__ == "__main__":
    main()
