"""Bi-PT deformation blocks.

``DeformBlockConcat`` is the full bidirectional cross-attention model;
``DeformBlockConcatSingleCA`` is the single-cross-attention (sCA) ablation. Both
fuse the atlas/SPC conditioning once, then integrate two NODE blocks (LADD) to
deform the atlas into an intermediate and a final mesh (deep supervision).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .node import ODEFuncConcat, ODEFuncConcatTranslation, NODEBlockConcat
from .cross_attention import SingleCrossAttentionModule, DoubleCrossAttentionModule


class DeformBlockConcat(nn.Module):
    def __init__(
        self,
        time=0.2,
        num_hidden=512,
        latent_len=32,
        tol=1e-5,
        tanh_dynamics=True,
        d_points_spc=5,
        d_points_atlas=5,   # will be 5 for 8-dim atlas, 6 for 9-dim atlas
        ca_nblocks=4,
        ca_nneighbor=16,
        norm_type="bn",
        cond_mode="concat",
        cond_norm="none",
        ca_dec_fc_norm_type: str = None,
        ca_dec_fp_norm_type: str = None,
        ode_force_fp32: bool = True,
        affine_dynamics: bool = True,
    ):
        super().__init__()
        self.time = float(time)
        self.latent_len = int(latent_len)

        self.cond_mode = str(cond_mode)
        self.cond_norm = str(cond_norm)

        self.d_points_spc = int(d_points_spc)
        self.d_points_atlas = int(d_points_atlas)

        self.doubleCA = DoubleCrossAttentionModule(
            d_model=num_hidden,
            d_points_q=self.d_points_atlas,
            d_points_x=self.d_points_spc,
            nblocks=ca_nblocks,
            nneighbor=ca_nneighbor,
            norm_type=norm_type,
            dec_fc_norm_type=ca_dec_fc_norm_type,
            dec_fp_norm_type=ca_dec_fp_norm_type,
        )

        self.cond_dim = self._infer_cond_dim(self.cond_mode, self.latent_len)
        self.cond_ln = nn.LayerNorm(self.cond_dim) if self.cond_norm == "layernorm" else None

        self.l1 = NODEBlockConcat(
            ODEFuncConcat(num_hidden=num_hidden, cond_dim=self.cond_dim, tanh_dynamics=tanh_dynamics),
            tol=tol,
            ode_force_fp32=ode_force_fp32,
        ) if affine_dynamics else NODEBlockConcat(ODEFuncConcatTranslation(num_hidden=num_hidden, cond_dim=self.cond_dim, tanh_dynamics=tanh_dynamics), tol=tol, ode_force_fp32=ode_force_fp32)
        self.l2 = NODEBlockConcat(
            ODEFuncConcat(num_hidden=num_hidden, cond_dim=self.cond_dim, tanh_dynamics=tanh_dynamics),
            tol=tol,
            ode_force_fp32=ode_force_fp32,
        ) if affine_dynamics else NODEBlockConcat(ODEFuncConcatTranslation(num_hidden=num_hidden, cond_dim=self.cond_dim, tanh_dynamics=tanh_dynamics), tol=tol, ode_force_fp32=ode_force_fp32)

    @staticmethod
    def _infer_cond_dim(cond_mode: str, L: int) -> int:
        if cond_mode == "concat":
            return 2 * L
        if cond_mode in {"mul", "add", "code", "global"}:
            return L
        raise ValueError(f"Unknown cond_mode: {cond_mode}")

    @staticmethod
    def _canon_global(g: torch.Tensor) -> torch.Tensor:
        if g.dim() == 2:
            return g
        if g.dim() == 3:
            return g[:, 0, :]
        raise ValueError(f"Unexpected code_global shape: {g.shape}")

    def _apply_cond_norm(self, cond: torch.Tensor) -> torch.Tensor:
        if self.cond_norm == "none":
            return cond
        if self.cond_norm == "layernorm":
            return self.cond_ln(cond)
        if self.cond_norm == "l2":
            return F.normalize(cond, p=2, dim=-1, eps=1e-8)
        raise ValueError(f"Unknown cond_norm: {self.cond_norm}")

    def _fuse_cond(self, code: torch.Tensor, code_global: torch.Tensor) -> torch.Tensor:
        B, N, L = code.shape
        g = code_global.unsqueeze(1).expand(B, N, L)

        m = self.cond_mode
        if m == "concat":
            cond = torch.cat([code, g], dim=-1)
        elif m == "mul":
            cond = code * g
        elif m == "add":
            cond = code + g
        elif m == "code":
            cond = code
        elif m == "global":
            cond = g
        else:
            raise ValueError(f"Unknown cond_mode: {m}")

        if cond.shape[-1] != self.cond_dim:
            raise RuntimeError(f"cond dim mismatch: got {cond.shape[-1]}, expected {self.cond_dim}")

        return self._apply_cond_norm(cond)

    def _build_state(self, xyz: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return torch.cat([xyz, cond], dim=-1)

    def forward(self, q_feat: torch.Tensor, spc: torch.Tensor, time: float = None):
        if time is None:
            time = self.time
        time = float(time)

        # Hard checks for your requirement:
        # spc has BN8 => feature_dim=8 => d_points=5
        if spc.shape[-1] != 3 + self.d_points_spc:
            raise ValueError(
                f"spc feature dim mismatch: got {spc.shape[-1]} but expected {3+self.d_points_spc} "
                f"(d_points_spc={self.d_points_spc})"
            )
        if q_feat.shape[-1] != 3 + self.d_points_atlas:
            raise ValueError(
                f"atlas feature dim mismatch: got {q_feat.shape[-1]} but expected {3+self.d_points_atlas} "
                f"(d_points_atlas={self.d_points_atlas})"
            )

        # ---- CA (before l1) ----
        code, code_global = self.doubleCA(q_feat, spc)
        code_global = self._canon_global(code_global)
        cond0 = self._fuse_cond(code, code_global)

        xyz0 = q_feat[..., :3]
        xz0 = self._build_state(xyz0, cond0)

        # ---- l1 ----
        xz1 = self.l1(xz0, time)
        xyz1 = xz1[..., :3]

        # ---- l2 ----
        xz2 = self.l2(xz1, time)
        return xyz1, xz2[..., :3]


class DeformBlockConcatSingleCA(nn.Module):
    """
    Same as DeformBlockConcat, but uses SingleCrossAttentionModule (q<-x) instead of DoubleCrossAttentionModule.

    Returns:
      (xyz1, xyz2)  where xyz1 is after l1, xyz2 after l2
    """
    def __init__(
        self,
        time=0.2,
        num_hidden=512,
        latent_len=32,
        tol=1e-5,
        tanh_dynamics=True,
        d_points_spc=5,
        d_points_atlas=5,
        ca_nblocks=4,
        ca_nneighbor=16,
        norm_type="bn",
        cond_mode="concat",
        cond_norm="none",
        ca_dec_fc_norm_type: str = None,
        ca_dec_fp_norm_type: str = None,
        ode_force_fp32: bool = True,
        affine_dynamics: bool = True,
    ):
        super().__init__()
        self.time = float(time)
        self.latent_len = int(latent_len)

        self.cond_mode = str(cond_mode)
        self.cond_norm = str(cond_norm)

        self.d_points_spc = int(d_points_spc)
        self.d_points_atlas = int(d_points_atlas)

        self.singleCA = SingleCrossAttentionModule(
            d_model=num_hidden,
            d_points_q=self.d_points_atlas,
            d_points_x=self.d_points_spc,
            nblocks=ca_nblocks,
            nneighbor=ca_nneighbor,
            norm_type=norm_type,
            dec_fc_norm_type=ca_dec_fc_norm_type,
            dec_fp_norm_type=ca_dec_fp_norm_type,
        )

        self.cond_dim = DeformBlockConcat._infer_cond_dim(self.cond_mode, self.latent_len)
        self.cond_ln = nn.LayerNorm(self.cond_dim) if self.cond_norm == "layernorm" else None

        self.l1 = NODEBlockConcat(
            ODEFuncConcat(num_hidden=num_hidden, cond_dim=self.cond_dim, tanh_dynamics=tanh_dynamics),
            tol=tol,
            ode_force_fp32=ode_force_fp32,
        ) if affine_dynamics else NODEBlockConcat(ODEFuncConcatTranslation(num_hidden=num_hidden, cond_dim=self.cond_dim, tanh_dynamics=tanh_dynamics), tol=tol, ode_force_fp32=ode_force_fp32)
        self.l2 = NODEBlockConcat(
            ODEFuncConcat(num_hidden=num_hidden, cond_dim=self.cond_dim, tanh_dynamics=tanh_dynamics),
            tol=tol,
            ode_force_fp32=ode_force_fp32,
        ) if affine_dynamics else NODEBlockConcat(ODEFuncConcatTranslation(num_hidden=num_hidden, cond_dim=self.cond_dim, tanh_dynamics=tanh_dynamics), tol=tol, ode_force_fp32=ode_force_fp32)

    @staticmethod
    def _canon_global(g: torch.Tensor) -> torch.Tensor:
        return DeformBlockConcat._canon_global(g)

    def _apply_cond_norm(self, cond: torch.Tensor) -> torch.Tensor:
        if self.cond_norm == "none":
            return cond
        if self.cond_norm == "layernorm":
            return self.cond_ln(cond)
        if self.cond_norm == "l2":
            return F.normalize(cond, p=2, dim=-1, eps=1e-8)
        raise ValueError(f"Unknown cond_norm: {self.cond_norm}")

    def _fuse_cond(self, code: torch.Tensor, code_global: torch.Tensor) -> torch.Tensor:
        B, N, L = code.shape
        g = code_global.unsqueeze(1).expand(B, N, L)

        m = self.cond_mode
        if m == "concat":
            cond = torch.cat([code, g], dim=-1)
        elif m == "mul":
            cond = code * g
        elif m == "add":
            cond = code + g
        elif m == "code":
            cond = code
        elif m == "global":
            cond = g
        else:
            raise ValueError(f"Unknown cond_mode: {m}")

        if cond.shape[-1] != self.cond_dim:
            raise RuntimeError(f"cond dim mismatch: got {cond.shape[-1]}, expected {self.cond_dim}")

        return self._apply_cond_norm(cond)

    @staticmethod
    def _build_state(xyz: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return torch.cat([xyz, cond], dim=-1)

    def forward(self, q_feat: torch.Tensor, spc: torch.Tensor, time: float = None):
        if time is None:
            time = self.time
        time = float(time)

        if spc.shape[-1] != 3 + self.d_points_spc:
            raise ValueError(
                f"spc feature dim mismatch: got {spc.shape[-1]} but expected {3+self.d_points_spc} "
                f"(d_points_spc={self.d_points_spc})"
            )
        if q_feat.shape[-1] != 3 + self.d_points_atlas:
            raise ValueError(
                f"atlas feature dim mismatch: got {q_feat.shape[-1]} but expected {3+self.d_points_atlas} "
                f"(d_points_atlas={self.d_points_atlas})"
            )

        # ---- single CA (before l1) ----
        code, code_global = self.singleCA(q_feat, spc)
        code_global = self._canon_global(code_global)
        cond0 = self._fuse_cond(code, code_global)

        xyz0 = q_feat[..., :3]
        xz0 = self._build_state(xyz0, cond0)

        # ---- l1 ----
        xz1 = self.l1(xz0, time)
        xyz1 = xz1[..., :3]

        # ---- l2 ----
        xz2 = self.l2(xz1, time)
        return xyz1, xz2[..., :3]



__all__ = [
    "DeformBlockConcat",
    "DeformBlockConcatSingleCA",
]
