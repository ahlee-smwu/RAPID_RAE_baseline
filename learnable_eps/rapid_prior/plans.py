"""GMM-blended interpolant paths, as free functions over a baseline ICPlan.

These are deliberately NOT methods on ``src/stage2/transport/path.py``'s
ICPlan: the baseline stays byte-identical so the prior=off control arm is
exactly the published code. Each function takes the plan object it operates
on as its first argument.

TIME-AXIS NOTE -- the single most important detail of the port.

RAPID / LightningDiT interpolate as
      alpha_t = t, sigma_t = 1 - t   ->  NOISE at t = 0,
      xt = t * x1 + (1 - t) * x0,   ut = x1 - x0,
with GMM schedule w(s) = q0 * exp(-decay_alpha * s).

THIS repository is reversed (see ICPlan.compute_alpha_t / compute_sigma_t):
      alpha_t = 1 - t, sigma_t = t   ->  NOISE at t = 1,
      xt = (1 - t) * x1 + t * x0,   ut = x0 - x1.

Substituting s = 1 - t maps one onto the other, so the schedule becomes
      w(t) = q0 * exp(-decay_alpha * (1 - t)),
peaking at w = q0 on the NOISE end (t = 1) and decaying toward the data end.
Copying w(s) verbatim would inject the GMM into the DATA end -- the mirror
image of the intent. verify_port.py check 3 pins this numerically.
"""

from __future__ import annotations

import torch as th


def expand_t_like_x(t, x):
    """Reshape t to broadcast against x. Mirrors path.expand_t_like_x."""
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


def gmm_weight_t(t, x, q0=0.5, decay_alpha=1.0, schedule="exp"):
    """GMM injection ratio w(t), broadcast to the shape of ``x``.

    schedule="exp"   -> w(t) = q0 * exp(-decay_alpha * (1 - t))   [w(1) = q0]
    schedule="const" -> w(t) = q0                                 [t-independent]
    """
    t_expand = expand_t_like_x(t, x)
    if schedule == "const":
        return th.full_like(t_expand, float(q0))
    if schedule != "exp":
        raise NotImplementedError(f"Unknown GMM schedule {schedule!r}")
    return q0 * th.exp(-decay_alpha * (1.0 - t_expand))


def plan_gmm_adaptive(plan, t, x0_gmm, x1, q0=0.5, decay_alpha=1.0, eps=None):
    """Time-adaptive GMM prior on this repo's reversed time axis.

    w(t)       = q0 * exp(-decay_alpha * (1 - t))   # w(t=1, noise end) = q0
    x0_blended = w * x0_gmm + (1 - w) * eps
    xt         = (1 - t) * x1 + t * x0_blended
    ut         = x0_blended - x1

    ``plan`` is the baseline ICPlan instance (unused here beyond signature
    symmetry with the const variant, but kept so both read the same way).
    ``eps`` may be supplied so the caller reuses the standard-Gaussian draw
    it already made -- at 196608-D a redundant randn is a real cost.

    NOTE: ut omits the ``t * dw/dt * (x0_gmm - eps)`` term, so it is the
    *approximate* conditional velocity -- the same approximation RAPID's
    plan_gmm_adaptive makes. Use plan_gmm_const for the exact-velocity control.
    """
    t_expand = expand_t_like_x(t, x1)
    w = gmm_weight_t(t, x1, q0=q0, decay_alpha=decay_alpha, schedule="exp")

    if eps is None:
        eps = th.randn_like(x1)
    x0_blended = w * x0_gmm + (1.0 - w) * eps

    xt = (1.0 - t_expand) * x1 + t_expand * x0_blended
    ut = x0_blended - x1
    return t, xt, ut


def plan_gmm_const(plan, t, x0_gmm, x1, q_const=0.5, eps=None):
    """Control arm: x0_blended does not depend on t.

    Because w is constant, x0_blended is a genuine t-independent endpoint and
    ut = x0_blended - x1 is the EXACT conditional velocity -- no missing dw/dt
    term. This isolates how much of any gain comes from the adaptive schedule
    versus from the GMM prior itself.
    """
    t_expand = expand_t_like_x(t, x1)
    w = gmm_weight_t(t, x1, q0=q_const, schedule="const")

    if eps is None:
        eps = th.randn_like(x1)
    x0_blended = w * x0_gmm + (1.0 - w) * eps

    xt = (1.0 - t_expand) * x1 + t_expand * x0_blended
    ut = x0_blended - x1
    return t, xt, ut
