import torch as th
import numpy as np
from functools import partial

def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


#################### Coupling Plans ####################

class ICPlan:
    """Linear Coupling Plan"""
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return 1 - t, -1
    
    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return t, 1
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return -1 / (1 - t)

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t

        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            # "linear": norm * (1 - t),
            "linear": norm * t,
            "decreasing": 0.25 * (norm * th.cos(np.pi * (1 - t)) + 1) ** 2,
            "inccreasing-decreasing": norm * th.sin(np.pi * (1 - t)) ** 2,
        }

        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        
        return diffusion

    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score
    
    def get_noise_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to denoiser
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Wrapper function: transfrom score prediction model to velocity
        Args:
            score: [batch_dim, ...] shaped tensor; score model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0
    
    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt
    
    def compute_ut(self, t, x0, x1, xt):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0
    
    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut

    # ------------------------------------------------------------------
    # RAPID adaptive prior (ported from LightningDiT/RAPID).
    #
    # TIME-AXIS NOTE -- this is the single most important detail of the port.
    # In RAPID/LightningDiT the interpolant is
    #       alpha_t = t, sigma_t = 1 - t   ->  NOISE lives at t = 0,
    #       xt = t * x1 + (1 - t) * x0,   ut = x1 - x0,
    # and the GMM injection schedule is w(s) = q0 * exp(-decay_alpha * s).
    #
    # In THIS repository the interpolant is reversed (see compute_alpha_t /
    # compute_sigma_t above):
    #       alpha_t = 1 - t, sigma_t = t   ->  NOISE lives at t = 1,
    #       xt = (1 - t) * x1 + t * x0,   ut = x0 - x1.
    #
    # Substituting s = 1 - t maps one convention onto the other, so the
    # schedule must become
    #       w(t) = q0 * exp(-decay_alpha * (1 - t)),
    # which peaks at w = q0 on the NOISE end (t = 1) and decays toward the
    # data end (t = 0), exactly as in RAPID. Copying w(s) = q0*exp(-a*s)
    # verbatim would inject the GMM into the DATA end instead -- the mirror
    # image of the intended behaviour. tools/verify_port.py check 3 pins
    # this equivalence numerically.
    # ------------------------------------------------------------------

    def gmm_weight_t(self, t, x, q0=0.5, decay_alpha=1.0, schedule="exp"):
        """GMM injection ratio w(t), broadcast to the shape of ``x``.

        schedule="exp"   -> w(t) = q0 * exp(-decay_alpha * (1 - t))   [w(1) = q0]
        schedule="const" -> w(t) = q0                                 [t-independent]
        """
        t_expand = expand_t_like_x(t, x)
        if schedule == "const":
            return th.full_like(t_expand, float(q0))
        if schedule != "exp":
            raise NotImplementedError(f"Unknown GMM schedule {schedule}")
        return q0 * th.exp(-decay_alpha * (1.0 - t_expand))

    def plan_gmm_adaptive(self, t, x0_gmm, x1, q0=0.5, decay_alpha=1.0, eps=None):
        """Time-adaptive GMM prior on this repo's reversed time axis.

        w(t)       = q0 * exp(-decay_alpha * (1 - t))   # w(t=1, noise end) = q0
        x0_blended = w * x0_gmm + (1 - w) * eps
        xt         = (1 - t) * x1 + t * x0_blended
        ut         = x0_blended - x1

        ``eps`` may be supplied so the caller can reuse the standard-Gaussian
        draw it already made (avoids a second randn of a 196608-D tensor).

        NOTE: ut omits the dw/dt term, so it is the *approximate* conditional
        velocity -- identical to the approximation RAPID's plan_gmm_adaptive
        makes. Use plan_gmm_const for the exact-velocity control.
        """
        t_expand = expand_t_like_x(t, x1)
        w = self.gmm_weight_t(t, x1, q0=q0, decay_alpha=decay_alpha, schedule="exp")

        if eps is None:
            eps = th.randn_like(x1)
        x0_blended = w * x0_gmm + (1.0 - w) * eps

        xt = (1.0 - t_expand) * x1 + t_expand * x0_blended
        ut = x0_blended - x1
        return t, xt, ut

    def plan_gmm_const(self, t, x0_gmm, x1, q_const=0.5, eps=None):
        """Control arm: x0_blended does not depend on t.

        Because w is constant, x0_blended is a genuine t-independent endpoint
        and ut = x0_blended - x1 is the EXACT conditional velocity -- there is
        no missing dw/dt term. This is the ablation that isolates how much of
        any gain comes from the adaptive schedule versus from the GMM prior
        itself.
        """
        t_expand = expand_t_like_x(t, x1)
        w = self.gmm_weight_t(t, x1, q0=q_const, schedule="const")

        if eps is None:
            eps = th.randn_like(x1)
        x0_blended = w * x0_gmm + (1.0 - w) * eps

        xt = (1.0 - t_expand) * x1 + t_expand * x0_blended
        ut = x0_blended - x1
        return t, xt, ut


class VPCPlan(ICPlan):
    """class for VP path flow matching"""

    def __init__(self, sigma_min=0.1, sigma_max=20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = lambda t: -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * (1 - t) * self.sigma_min 
        self.d_log_mean_coeff = lambda t: 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min

    def compute_alpha_t(self, t):
        """Compute coefficient of x0"""
        p_alpha_t = 2 * self.log_mean_coeff(t)
        alpha_t = th.sqrt(1 - th.exp(p_alpha_t))
        d_alpha_t = th.exp(p_alpha_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * alpha_t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        """Compute coefficient of x1"""
        sigma_t = self.log_mean_coeff(t)
        sigma_t = th.exp(sigma_t)
        d_sigma_t = sigma_t * self.d_log_mean_coeff(t)
        return sigma_t, d_sigma_t
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        return d_alpha_t / alpha_t

    def compute_drift(self, x, t):
        """Compute the drift term of the SDE"""
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2
    

class GVPCPlan(ICPlan):
    def __init__(self, sigma=0.0):
        super().__init__(sigma)
    
    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = th.cos(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * th.sin(t * np.pi / 2)
        return alpha_t, d_alpha_t
    
    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        sigma_t = th.sin(t * np.pi / 2)
        d_sigma_t = np.pi / 2 * th.cos(t * np.pi / 2)
        return sigma_t, d_sigma_t
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return -np.pi / 2 * th.tan(t * np.pi / 2)