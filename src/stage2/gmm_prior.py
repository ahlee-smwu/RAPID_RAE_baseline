"""RAPID adaptive GMM prior for RAE (DINOv2) stage-2 latents.

Ported from the LightningDiT/VA-VAE implementation in the RAPID repository
(``adaptive-prior/train.py`` : ``get_cluster_gmm`` / ``precompute_sigma_scale`` /
``lowpass_avg``, and ``adaptive-prior/inference.py`` : the z-blending block).

Why this is not a copy-paste of the original
--------------------------------------------
RAE latents are 768x16x16 = 196,608-D, 24x larger than VA-VAE's 32x16x16 =
8,192-D. The original fitting artefacts do not fit on disk at that width:

    per-class PCA bases : 1000 x 256 x 196608 x 4B = 201 GB
    data-space diag var : 1000 x K x 196608 x 4B   =  15.7 GB (K=20)

So this port changes the storage layout while keeping the arithmetic
identical:

  * ONE shared PCA basis ``U`` of shape (d, D) for all classes  -> 201 MB
  * per-class GMM (means/diag-cov/weights) lives in PCA space   ->  40 MB
  * data-space diagonal variances are RESTORED ON THE FLY as ``v_k @ U**2``,
    which is exactly the value the original read out of its precomputed
    table (see ``precompute_sigma_scale``: ``diag_var_all = v_safe @ U2``)
  * cluster MEANS stay in full space as an fp16 memmap (K=10 -> 3.9 GB).
    Means are the one thing that must not be compressed: projecting 1000
    class means onto a 256-D shared basis would destroy the very class
    information the prior exists to supply.

Time axis
---------
This repository puts NOISE at t=1 (``alpha_t = 1-t``, ``sigma_t = t``), the
mirror image of RAPID. The blending schedule therefore reads
``w(t) = q0 * exp(-decay_alpha * (1-t))``; see the note in
``stage2/transport/path.py``. This module only builds ``x0_gmm``; the
schedule itself lives in the path sampler.
"""

from __future__ import annotations

import math
import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Low-pass filter on cluster means ("data-aware prior")
# ----------------------------------------------------------------------

def lowpass_avg(mu: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """3x3 box low-pass filter over the spatial grid, blended by ``alpha``.

    mu: (B, C, H, W). Channels are filtered independently so cross-channel
    structure is preserved; only the 16x16 spatial layout is smoothed.

    alpha = 0.0 -> untouched, alpha = 1.0 -> fully smoothed.
    Identical to RAPID's ``lowpass_avg``.
    """
    if alpha == 0.0:
        return mu
    B, C, H, W = mu.shape
    mu_2d = mu.reshape(B * C, 1, H, W)
    smoothed = F.avg_pool2d(mu_2d, kernel_size=3, stride=1, padding=1)
    smoothed = smoothed.reshape(B, C, H, W)
    return (1.0 - alpha) * mu + alpha * smoothed


def estimate_lpf_alpha_minus3db(kernel_size: int = 3) -> float:
    """DIAGNOSTIC ONLY -- do not wire this into training.

    Returns the blend alpha whose 3x3-box/identity mixture has half POWER
    (-3 dB) at the Nyquist spatial frequency.

    Caveat carried over from the hand-off notes: it was never confirmed
    whether the thesis derivation of alpha* = 1.008 for ImageNet defines the
    -3 dB point on a power ratio (R(f) = 0.5) or an amplitude ratio
    (|H(f)| = 1/sqrt(2)). The two differ by roughly a factor 0.7 in alpha.
    This implements the POWER convention. Training uses whatever
    ``prior.lpf_alpha`` the config specifies (1.0 for ImageNet), never this.
    """
    if kernel_size != 3:
        raise NotImplementedError("Only the 3x3 box filter is characterised.")
    # 1-D 3-tap box response at Nyquist (f = 0.5 cycles/sample):
    #   H_box(f) = (1 + 2*cos(2*pi*f)) / 3  ->  H_box(0.5) = -1/3
    # Separable in 2-D, so the 2-D response at the corner frequency is
    #   H2 = H_box(0.5)**2 = 1/9
    h_box = (1.0 + 2.0 * math.cos(math.pi)) / 3.0
    h2 = h_box ** 2
    # Blended filter: H(a) = (1 - a) + a * h2. Solve H(a)**2 = 0.5 (power).
    target = math.sqrt(0.5)
    return (1.0 - target) / (1.0 - h2)


# ----------------------------------------------------------------------
# GMM prior
# ----------------------------------------------------------------------

class GMMPrior:
    """Class-conditional GMM prior over RAE latents.

    Checkpoint layout (produced by ``src/fit_gmm_rae.py``)::

        {
          "pca_U":        (d, D) float32   shared PCA basis, orthonormal rows
          "pca_mean":     (D,)   float32   shared PCA centre
          "means_pca":    (C, K, d) float32
          "covs_pca":     (C, K, d) float32   diagonal variances in PCA space
          "weights":      (C, K)   float32   mixture weights, rows sum to 1
          "means_file":   str               basename of the fp16 memmap
          "means_shape":  (C, K, D)
          "latent_shape": (C_lat, H, W)
          "num_classes":  int
          "K":            int
          "global_sigma_scale": float
          "latent_mean":  float             diagnostics from the fit
          "latent_std":   float
        }

    ``means_file`` is a separate ``.npy`` fp16 memmap because at K=10 it is
    3.9 GB -- far too large to pickle.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        lpf_alpha: float = 1.0,
        use_weight: bool = True,
        stochastic_assign: bool = False,
        means_device: str = "mmap",
        eps: float = 1e-8,
        logger=None,
    ):
        self.device = device
        self.lpf_alpha = float(lpf_alpha)
        self.use_weight = bool(use_weight)
        self.stochastic_assign = bool(stochastic_assign)
        self.eps = float(eps)
        self.means_device = means_device
        self._logger = logger
        self._stats_logged = False

        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        self.num_classes = int(ckpt["num_classes"])
        self.K = int(ckpt["K"])
        self.latent_shape = tuple(int(v) for v in ckpt["latent_shape"])
        self.D = int(np.prod(self.latent_shape))
        self.global_sigma_scale = float(ckpt["global_sigma_scale"])

        # --- PCA-space tensors: small, always resident on GPU in fp32 ---
        self.U = torch.as_tensor(ckpt["pca_U"], dtype=torch.float32, device=device)
        self.pca_mean = torch.as_tensor(ckpt["pca_mean"], dtype=torch.float32, device=device)
        self.means_pca = torch.as_tensor(ckpt["means_pca"], dtype=torch.float32, device=device)
        self.covs_pca = torch.as_tensor(ckpt["covs_pca"], dtype=torch.float32, device=device)
        self.weights = torch.as_tensor(ckpt["weights"], dtype=torch.float32, device=device)
        self.d = self.U.shape[0]

        if self.U.shape[1] != self.D:
            raise ValueError(
                f"PCA basis has D={self.U.shape[1]} but latent_shape implies D={self.D}."
            )

        # U squared, used to restore data-space diagonal variances.
        self.U2 = self.U * self.U  # (d, D)

        # --- full-space cluster means ---
        means_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), ckpt["means_file"])
        means_shape = tuple(int(v) for v in ckpt["means_shape"])
        if not os.path.exists(means_path):
            raise FileNotFoundError(f"Cluster means file not found: {means_path}")

        if means_device == "mmap":
            # Reads B x D x 2B per batch. MUST live on local SSD -- on NFS this
            # becomes the training bottleneck.
            self._means_np = np.load(means_path, mmap_mode="r")
            self._means_t = None
        elif means_device == "cpu":
            # Beware: every rank on the node holds its own copy. Fine for 1-2
            # ranks, ruinous for 8.
            self._means_np = None
            self._means_t = torch.from_numpy(np.load(means_path)).to(torch.float16)
        elif means_device == "cuda":
            self._means_np = None
            self._means_t = torch.from_numpy(np.load(means_path)).to(device=device, dtype=torch.float16)
        else:
            raise ValueError(f"Unknown means_device {means_device!r} (mmap|cpu|cuda)")

        if self._means_t is not None and tuple(self._means_t.shape) != means_shape:
            raise ValueError(f"means tensor shape {tuple(self._means_t.shape)} != {means_shape}")

        self.latent_mean = float(ckpt.get("latent_mean", float("nan")))
        self.latent_std = float(ckpt.get("latent_std", float("nan")))

        self._log(
            f"[RAPID prior] classes={self.num_classes} K={self.K} pca_dim={self.d} "
            f"D={self.D} means_device={means_device} lpf_alpha={self.lpf_alpha} "
            f"use_weight={self.use_weight} stochastic_assign={self.stochastic_assign} "
            f"global_sigma_scale={self.global_sigma_scale:.6f} "
            f"fit_latent_mean={self.latent_mean:.4f} fit_latent_std={self.latent_std:.4f}"
        )

    # -- helpers -------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)
        else:
            print(msg)

    def _gather_means(self, y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """(B, D) float32 cluster means for the (class, cluster) pairs."""
        if self._means_t is not None:
            sel = self._means_t[y.to(self._means_t.device), k.to(self._means_t.device)]
            return sel.to(device=self.device, dtype=torch.float32)
        # memmap path: index on CPU, one contiguous row per sample
        y_np = y.detach().cpu().numpy()
        k_np = k.detach().cpu().numpy()
        sel = np.stack([self._means_np[int(c), int(j)] for c, j in zip(y_np, k_np)], axis=0)
        return torch.from_numpy(np.ascontiguousarray(sel)).to(device=self.device, dtype=torch.float32)

    def _restore_diag_var(self, v_pca: torch.Tensor) -> torch.Tensor:
        """PCA-space diagonal variances -> data-space diagonal variances.

        v_pca: (B, d) -> (B, D) via ``v @ U**2``.

        This is the same arithmetic the original performed when it BUILT its
        (C, K, D) table (``diag_var_all = v_safe @ U2``); we simply evaluate it
        per batch instead of storing 15.7 GB.
        """
        return torch.clamp(v_pca, min=self.eps) @ self.U2

    def _posterior(self, x_flat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Responsibility of each of the K clusters for each sample. (B, K)"""
        means_pca = self.means_pca[y]          # (B, K, d)
        v = torch.clamp(self.covs_pca[y], min=self.eps)  # (B, K, d)
        w = self.weights[y]                    # (B, K)

        # Project onto the shared PCA basis.
        x_pca = (x_flat - self.pca_mean) @ self.U.t()    # (B, d)

        diff = x_pca.unsqueeze(1) - means_pca            # (B, K, d)
        mahal = (diff * diff / v).sum(dim=2)             # (B, K)
        log_det = torch.log(v).sum(dim=2)                # (B, K)
        log_prob = -0.5 * (mahal + log_det)
        if self.use_weight:
            log_prob = log_prob + torch.log(w + self.eps)

        posterior = torch.softmax(log_prob, dim=1)
        if torch.isnan(posterior).any():
            posterior = torch.nan_to_num(posterior, nan=1.0 / posterior.shape[1])
        return posterior

    def _fp32(self):
        """Force fp32 regardless of any autocast the caller is inside.

        The PCA projection and the diagonal-variance restoration are matmuls
        against a (d, D) basis; under bf16 autocast they lose enough precision
        to distort the posterior and the recovered sigma. train.py already
        disables autocast around the training call, but the inference scripts
        build their latents inside a bf16 autocast block, so the guarantee
        belongs here rather than at each call site.
        """
        return torch.amp.autocast(device_type=self.device.type, enabled=False)

    def _compose(self, y: torch.Tensor, k: torch.Tensor, shape) -> torch.Tensor:
        """Build x0_gmm = LPF(mu) + sigma * randn for the chosen clusters."""
        B = y.shape[0]
        mu = self._gather_means(y, k)                       # (B, D)
        v_pca = self.covs_pca[y, k]                         # (B, d)
        diag_var = self._restore_diag_var(v_pca)            # (B, D)
        sigma = torch.sqrt(diag_var + self.eps) * self.global_sigma_scale

        mu = mu.view(B, *shape)
        sigma = sigma.view(B, *shape)
        mu_lpf = lowpass_avg(mu, alpha=self.lpf_alpha)

        x0_gmm = mu_lpf + sigma * torch.randn_like(mu_lpf)
        return x0_gmm, mu_lpf

    # -- public API ----------------------------------------------------

    @torch.no_grad()
    def build_x0_gmm(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """TRAINING: posterior-assign each latent to a cluster, then draw.

        z: (B, C, H, W) the exact tensor the diffusion model is trained on
        y: (B,) class labels

        Runs entirely in fp32 -- the caller is responsible for disabling
        autocast (PCA projection and diagonal-variance restoration are not
        numerically safe in bf16).
        """
        with self._fp32():
            z = z.float()
            B = z.shape[0]
            shape = tuple(z.shape[1:])
            x_flat = z.reshape(B, -1)

            posterior = self._posterior(x_flat, y)
            if self.stochastic_assign:
                k = torch.multinomial(posterior, num_samples=1).squeeze(1)
            else:
                k = posterior.argmax(dim=1)

            x0_gmm, mu_lpf = self._compose(y, k, shape)
            self._maybe_log_stats(mu_lpf, x0_gmm)
            return x0_gmm

    @torch.no_grad()
    def sample_x0_gmm(self, y: torch.Tensor, shape=None) -> torch.Tensor:
        """INFERENCE: draw a cluster from the class's mixture weights.

        No latent is available at sampling time, so the cluster is drawn from
        the prior mixture weights rather than a posterior.
        """
        if shape is None:
            shape = self.latent_shape
        with self._fp32():
            w = self.weights[y]                              # (B, K)
            k = torch.multinomial(w, num_samples=1).squeeze(1)
            x0_gmm, _ = self._compose(y, k, tuple(shape))
            return x0_gmm

    @torch.no_grad()
    def init_latent(self, y: torch.Tensor, shape=None, q0: float = 0.5) -> torch.Tensor:
        """INFERENCE: the initial latent at the noise end (t = 1).

        Must mirror the training blend evaluated at t=1, where
        ``w(1) = q0 * exp(0) = q0``::

            z_init = q0 * x0_gmm + (1 - q0) * eps

        NEVER renormalise the result (``z = z / z.std()``). Doing so silently
        cancels the GMM/eps ratio the model was trained on; it was the direct
        cause of the 13th-15th experiment failures.
        """
        if shape is None:
            shape = self.latent_shape
        with self._fp32():
            x0_gmm = self.sample_x0_gmm(y, shape=shape)
            eps = torch.randn_like(x0_gmm)
            z = q0 * x0_gmm + (1.0 - q0) * eps
            self._maybe_log_init(z)
            return z

    # -- one-shot diagnostics -----------------------------------------

    def _maybe_log_stats(self, mu_lpf: torch.Tensor, x0_gmm: torch.Tensor) -> None:
        if self._stats_logged:
            return
        self._stats_logged = True
        self._log(
            f"[RAPID prior] var(mu_lpf)={mu_lpf.var().item():.4f} "
            f"var(x0_gmm)={x0_gmm.var().item():.4f}"
        )

    def _maybe_log_init(self, z: torch.Tensor) -> None:
        if getattr(self, "_init_logged", False):
            return
        self._init_logged = True
        # Record this number: the baseline N(0, I) has var = 1.0, so a value
        # near 0.5 means the initial latent carries half the usual variance.
        # Training and inference share the configuration so training is
        # consistent, but the effective SNR differs from baseline and the
        # value belongs in the paper. Below ~0.3, revisit q0.
        self._log(f"[RAPID prior] var(z_init)={z.var().item():.4f}")


# ----------------------------------------------------------------------
# Sampler wrapper
# ----------------------------------------------------------------------

def wrap_sampler_with_prior(
    sample_fn,
    prior: Optional[GMMPrior],
    q0: float = 0.5,
    num_classes: int = 1000,
    null_label: Optional[int] = None,
    logger=None,
):
    """Wrap a Sampler-produced ``sample_fn`` so it starts from the GMM prior.

    ``sample_fn(init, model, **model_kwargs)`` integrates from the noise end
    (t=1) to the data end (t=0), so replacing ``init`` is the whole change --
    ``src/eval/__init__.py`` needs no modification.

    The class labels are read from ``model_kwargs['y']``, which
    ``evaluate_generation_distributed`` already supplies. Under
    classifier-free guidance ``y`` arrives as ``[y_real; y_null]`` and ``init``
    is the duplicated ``[z; z]``; the wrapper detects this and builds one
    prior draw for the real half, then duplicates it so both branches start
    from the same latent -- which is what the unwrapped code does too.
    """
    if prior is None:
        return sample_fn

    if null_label is None:
        null_label = num_classes

    def _wrapped(init, model, **model_kwargs):
        y = model_kwargs.get('y', None)
        if y is None:
            if logger is not None:
                logger.warning(
                    "[RAPID prior] sampler called without 'y'; drawing labels "
                    "uniformly. Check that the eval path passes class labels."
                )
            y = torch.randint(0, num_classes, (init.shape[0],), device=init.device)

        shape = tuple(init.shape[1:])
        n_init = init.shape[0]
        half = n_init // 2

        # CFG layout: init is [z; z] and y is [y_real; y_null]. Detect it from
        # the labels (cheap, exact) rather than by comparing the two halves of
        # a 196608-D tensor. Real labels are 0..num_classes-1, so a trailing
        # all-null half is unambiguous.
        is_cfg = (
            n_init % 2 == 0
            and n_init > 0
            and y.shape[0] == n_init
            and bool((y[half:] == null_label).all())
        )

        if is_cfg:
            z = prior.init_latent(y[:half], shape=shape, q0=q0)
            z = torch.cat([z, z], dim=0)
        else:
            z = prior.init_latent(y[:n_init], shape=shape, q0=q0)

        return sample_fn(z.to(dtype=init.dtype), model, **model_kwargs)

    return _wrapped
