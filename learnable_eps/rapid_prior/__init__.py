"""RAPID adaptive GMM prior -- experiment code layered on the RAE baseline.

Everything here is additive: src/ is byte-identical to the published
baseline, so the prior=off control arm is exactly the baseline code.
"""

from .gmm_prior import (
    GMMPrior,
    wrap_sampler_with_prior,
    lowpass_avg,
    estimate_lpf_alpha_minus3db,
)
from .plans import gmm_weight_t, plan_gmm_adaptive, plan_gmm_const
from .losses import training_losses_gmm

__all__ = [
    "GMMPrior",
    "wrap_sampler_with_prior",
    "lowpass_avg",
    "estimate_lpf_alpha_minus3db",
    "gmm_weight_t",
    "plan_gmm_adaptive",
    "plan_gmm_const",
    "training_losses_gmm",
]
