"""Training loss on the GMM-blended adaptive prior path.

A free function over a baseline ``Transport`` instance, so
``src/stage2/transport/transport.py`` stays byte-identical.
"""

from __future__ import annotations

import torch as th

from stage2.transport.transport import ModelType
from stage2.transport.utils import mean_flat

from .plans import plan_gmm_adaptive, plan_gmm_const


def training_losses_gmm(
    transport,
    model,
    x1,
    x0_gmm,
    model_kwargs=None,
    q0=0.5,
    decay_alpha=1.0,
    schedule="exp",
):
    """Velocity loss on the GMM-blended path.

    Args:
    - transport: the baseline Transport instance (supplies sample() and path_sampler)
    - x1: datapoint (the RAE latent the model is trained on)
    - x0_gmm: per-sample GMM prior draw, same shape as x1
    - q0 / decay_alpha / schedule: see plans.gmm_weight_t

    The standard-Gaussian draw produced by ``transport.sample`` is REUSED as
    the blend's eps rather than drawing a second one.
    """
    if transport.model_type != ModelType.VELOCITY:
        raise NotImplementedError(
            "training_losses_gmm supports velocity prediction only; "
            f"got model_type={transport.model_type}."
        )
    if schedule not in ("exp", "const"):
        # Do not fall back silently: a typo'd schedule would otherwise run a
        # different experiment than the config claims.
        raise NotImplementedError(
            f"Unknown GMM schedule {schedule!r}; expected 'exp' or 'const'."
        )

    if model_kwargs is None:
        model_kwargs = {}

    t, x0, x1 = transport.sample(x1)

    if schedule == "const":
        t, xt, ut = plan_gmm_const(
            transport.path_sampler, t, x0_gmm, x1, q_const=q0, eps=x0
        )
    else:
        t, xt, ut = plan_gmm_adaptive(
            transport.path_sampler, t, x0_gmm, x1,
            q0=q0, decay_alpha=decay_alpha, eps=x0,
        )

    model_output = model(xt, t, **model_kwargs)
    B, *_, C = xt.shape
    assert model_output.size() == (B, *xt.size()[1:-1], C)

    terms = {}
    terms['pred'] = model_output
    terms['t'] = t
    terms['loss'] = mean_flat(((model_output - ut) ** 2))
    return terms
