"""Neural-ODE deformation dynamics for Bi-PT (LADD).

``ODEFuncConcat`` is the per-point locally-affine + translation velocity field
that realises the Locally Affine Diffeomorphic Deformation (LADD, paper Sec. 2.3);
``ODEFuncConcatTranslation`` is the translation-only ablation; ``NODEBlockConcat``
integrates the field with torchdiffeq (FP32 solve by default for stability).
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class ODEFuncConcat(nn.Module):
    """
    Fixed-dim concat-state ODE RHS:
      state xz = [xyz | cond]
    - xyz dynamics learned
    - cond dynamics frozen (zero)
    - gating: point_features * shape_features (unchanged)
    """
    def __init__(self, num_hidden=512, cond_dim=1024, tanh_dynamics=True):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.ifTanhDynamics = bool(tanh_dynamics)

        self.l1 = nn.Linear(3, num_hidden)
        self.cond = nn.Linear(self.cond_dim, num_hidden)

        self.l2 = nn.Linear(num_hidden, num_hidden)
        self.l3 = nn.Linear(num_hidden, num_hidden)
        self.l4 = nn.Linear(num_hidden, 3)

        self.a1 = nn.Linear(num_hidden, 9)
        self.a2 = nn.Linear(9, 9)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.nfe = 0
        self.register_buffer("_zeros_cond", torch.zeros((1, 1, self.cond_dim)), persistent=False)

    def forward(self, t, xz):
        B, N, D = xz.shape
        expected = 3 + self.cond_dim
        if D != expected:
            raise ValueError(f"ODEFuncConcat expects last dim {expected}, got {D}. xz.shape={tuple(xz.shape)}")

        xyz = xz[..., :3]
        cond = xz[..., 3:]

        point_features = self.relu(self.l1(xyz))
        shape_features = self.tanh(self.cond(cond))

        h = point_features * shape_features
        h = self.relu(self.l2(h)) + h
        h = self.relu(self.l3(h)) + h

        A = self.a1(h)
        if self.ifTanhDynamics:
            A = self.tanh(A)
        A = self.a2(A).view(B, N, 3, 3)

        transformed = torch.einsum("bnij,bnj->bni", A, xyz)

        dyn = self.l4(h)
        if self.ifTanhDynamics:
            dyn = self.tanh(dyn)

        dxyz = transformed + dyn
        self.nfe += 1

        # frozen cond dynamics (zeros)
        z0 = self._zeros_cond
        if z0.dtype != dxyz.dtype:
            z0 = z0.to(dtype=dxyz.dtype)
        dcond = z0.expand(B, N, -1)

        return torch.cat([dxyz, dcond], dim=-1)


class ODEFuncConcatTranslation(nn.Module):
    """
    Fixed-dim concat-state ODE RHS:
      state xz = [xyz | cond]
    - xyz dynamics learned
    - cond dynamics frozen (zero)
    - gating: point_features * shape_features (unchanged)
    """
    def __init__(self, num_hidden=512, cond_dim=1024, tanh_dynamics=True):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.ifTanhDynamics = bool(tanh_dynamics)

        self.l1 = nn.Linear(3, num_hidden)
        self.cond = nn.Linear(self.cond_dim, num_hidden)

        self.l2 = nn.Linear(num_hidden, num_hidden)
        self.l3 = nn.Linear(num_hidden, num_hidden)
        self.l4 = nn.Linear(num_hidden, 3)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.nfe = 0
        self.register_buffer("_zeros_cond", torch.zeros((1, 1, self.cond_dim)), persistent=False)

    def forward(self, t, xz):
        B, N, D = xz.shape
        expected = 3 + self.cond_dim
        if D != expected:
            raise ValueError(f"ODEFuncConcat expects last dim {expected}, got {D}. xz.shape={tuple(xz.shape)}")

        xyz = xz[..., :3]
        cond = xz[..., 3:]

        point_features = self.relu(self.l1(xyz))
        shape_features = self.tanh(self.cond(cond))

        h = point_features * shape_features
        h = self.relu(self.l2(h)) + h
        h = self.relu(self.l3(h)) + h

        dyn = self.l4(h)
        if self.ifTanhDynamics:
            dyn = self.tanh(dyn)

        dxyz = dyn
        self.nfe += 1

        # frozen cond dynamics (zeros)
        z0 = self._zeros_cond
        if z0.dtype != dxyz.dtype:
            z0 = z0.to(dtype=dxyz.dtype)
        dcond = z0.expand(B, N, -1)

        return torch.cat([dxyz, dcond], dim=-1)


class NODEBlockConcat(nn.Module):
    """
    ODE solver for the concat-state Neural ODE used by Bi-PT.

    The integration state is ``[xyz | cond]``: the point coordinates are advanced
    by the learned velocity field while the conditioning code is carried along
    unchanged (zero dynamics). Supports two solver modes:
      - adaptive: default dopri5-style integration through torchdiffeq
      - fixed-step: e.g. rk4/euler with an explicit ``step_size``

    The ODE solve runs in FP32 by default (``ode_force_fp32=True``) to avoid
    step-size underflow under mixed-precision training.
    """
    def __init__(
        self,
        odefunc: ODEFuncConcat,
        tol=1e-5,
        ode_force_fp32: bool = True,
        fixed_step: bool = False,
        step_size: float | None = None,
        method_adaptive: str = "dopri5",
        method_fixed: str = "rk4",
    ):
        super().__init__()
        self.odefunc = odefunc
        self.rtol = float(tol)
        self.atol = float(tol)
        self.cost = 0
        self.ode_force_fp32 = bool(ode_force_fp32)
        self.fixed_step = bool(fixed_step)
        self.step_size = None if step_size is None else float(step_size)
        self.method_adaptive = str(method_adaptive)
        self.method_fixed = str(method_fixed)

    def _solve(self, xz: torch.Tensor, t: torch.Tensor, fixed_step: bool, step_size: float | None):
        if fixed_step:
            dt = float(step_size) if step_size is not None else abs(float(t[-1].item() - t[0].item()))
            if dt <= 0:
                raise ValueError(f"fixed-step NODEBlockConcat requires positive step_size, got {dt}")
            out = odeint(
                self.odefunc,
                xz,
                t,
                method=self.method_fixed,
                options={"step_size": dt},
            )
        else:
            out = odeint(
                self.odefunc,
                xz,
                t,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method_adaptive,
            )
        self.cost = self.odefunc.nfe
        return out

    def _ode_solve_fp32(
        self,
        xz: torch.Tensor,
        t: torch.Tensor,
        fixed_step: bool,
        step_size: float | None,
    ) -> torch.Tensor:
        # Disable autocast so ODE stays FP32 even under AMP.
        with torch.cuda.amp.autocast(enabled=False):
            xz32 = xz.float()
            t32 = t.to(dtype=torch.float32)
            out = self._solve(xz32, t32, fixed_step=fixed_step, step_size=step_size)
            return out[-1].to(dtype=xz.dtype)

    def forward(self, xz, time: float, fixed_step: bool = None, step_size: float = None):
        self.odefunc.nfe = 0
        if fixed_step is None:
            fixed_step = self.fixed_step
        if step_size is None:
            step_size = self.step_size

        t = torch.tensor([0.0, float(time)], device=xz.device, dtype=torch.float32)

        if self.ode_force_fp32:
            return self._ode_solve_fp32(xz, t, fixed_step=bool(fixed_step), step_size=step_size)

        out = self._solve(
            xz,
            t.to(dtype=xz.dtype),
            fixed_step=bool(fixed_step),
            step_size=step_size,
        )
        return out[-1]

    def invert(self, xz, time: float, fixed_step: bool = None, step_size: float = None):
        self.odefunc.nfe = 0
        if fixed_step is None:
            fixed_step = self.fixed_step
        if step_size is None:
            step_size = self.step_size

        t = torch.tensor([float(time), 0.0], device=xz.device, dtype=torch.float32)

        if self.ode_force_fp32:
            return self._ode_solve_fp32(xz, t, fixed_step=bool(fixed_step), step_size=step_size)

        out = self._solve(
            xz,
            t.to(dtype=xz.dtype),
            fixed_step=bool(fixed_step),
            step_size=step_size,
        )
        return out[-1]



__all__ = [
    "ODEFuncConcat",
    "ODEFuncConcatTranslation",
    "NODEBlockConcat",
]
