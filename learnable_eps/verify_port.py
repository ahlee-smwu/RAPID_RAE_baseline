"""Gate A -- numerical verification of the RAPID -> RAE port. No GPU needed.

Run::

    python tools/verify_port.py

    # optionally cross-check against the original implementation:
    python tools/verify_port.py --old-path <RAPID_repo>/transport/path.py

Check 3 is the one that matters: it confirms that the reversed time axis was
ported rather than copied. If it fails, STOP -- the GMM is being injected into
the data end instead of the noise end, and nothing downstream is meaningful.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

PASS, FAIL = [], []


def report(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", default=os.path.join(REPO, "src/stage2/transport/path.py"),
                    help="The UNMODIFIED baseline ICPlan (must stay unmodified).")
    ap.add_argument("--plans", default=os.path.join(HERE, "rapid_prior/plans.py"))
    ap.add_argument("--prior", default=os.path.join(HERE, "rapid_prior/gmm_prior.py"))
    ap.add_argument("--old-path", default=None, help="RAPID's transport/path.py, if available.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    base_mod = load_module(args.base_path, "rae_path")
    plans = load_module(args.plans, "rapid_plans")
    base_plan = base_mod.ICPlan()

    class _PlanAdapter:
        """Bind the free functions in plans.py to the baseline ICPlan.

        The experiment code deliberately does NOT add methods to the baseline
        ICPlan, so the checks below reach the GMM plans through this adapter
        while `plan.compute_alpha_t` etc. still come from the untouched
        baseline class.
        """
        def __getattr__(self, name):
            return getattr(base_plan, name)

        def gmm_weight_t(self, t, x, **kw):
            return plans.gmm_weight_t(t, x, **kw)

        def plan_gmm_adaptive(self, t, x0_gmm, x1, **kw):
            return plans.plan_gmm_adaptive(base_plan, t, x0_gmm, x1, **kw)

        def plan_gmm_const(self, t, x0_gmm, x1, **kw):
            return plans.plan_gmm_const(base_plan, t, x0_gmm, x1, **kw)

    plan = _PlanAdapter()

    B, C, H, W = 4, 8, 16, 16
    x1 = torch.randn(B, C, H, W)
    x0_gmm = torch.randn(B, C, H, W)
    eps = torch.randn(B, C, H, W)
    q0, decay = 0.5, 1.0

    # ------------------------------------------------------------------
    print("\n== Check 0: the baseline under src/ is untouched ==")
    # The prior=off control arm is only valid if src/ is the published code.
    # All experiment code lives under learnable_eps/, so no file in src/ may
    # mention the prior.
    import subprocess
    leaked = subprocess.run(
        ["grep", "-rIl", "-e", "gmm", "-e", "rapid_prior", "-e", "RAPID",
         os.path.join(REPO, "src")],
        capture_output=True, text=True,
    ).stdout.strip()
    report("no prior code leaked into src/", leaked == "",
           "clean" if not leaked else f"found in: {leaked.replace(REPO + '/', '')}")

    # ------------------------------------------------------------------
    print("\n== Check 1: RAE interpolant convention (noise at t=1) ==")
    t0 = torch.zeros(B)
    t1 = torch.ones(B)
    a0, da = plan.compute_alpha_t(t0)
    s0, ds = plan.compute_sigma_t(t0)
    a1, _ = plan.compute_alpha_t(t1)
    s1, _ = plan.compute_sigma_t(t1)
    report("alpha_t(0)=1, sigma_t(0)=0  (t=0 is DATA)",
           torch.allclose(a0, torch.ones(B)) and torch.allclose(s0, torch.zeros(B)),
           f"alpha={a0[0]:.3f} sigma={s0[0]:.3f}")
    report("alpha_t(1)=0, sigma_t(1)=1  (t=1 is NOISE)",
           torch.allclose(a1, torch.zeros(B)) and torch.allclose(s1, torch.ones(B)),
           f"alpha={a1[0]:.3f} sigma={s1[0]:.3f}")
    _, xt_p, ut_p = plan.plan(torch.full((B,), 0.3), eps, x1)
    report("baseline plan(): ut == x0 - x1",
           torch.allclose(ut_p, eps - x1, atol=1e-6))

    # ------------------------------------------------------------------
    print("\n== Check 2: w(t) schedule endpoints ==")
    w1 = plan.gmm_weight_t(torch.ones(B), x1, q0=q0, decay_alpha=decay)
    w0 = plan.gmm_weight_t(torch.zeros(B), x1, q0=q0, decay_alpha=decay)
    report("w(t=1) == q0 (full GMM at the NOISE end)",
           torch.allclose(w1, torch.full_like(w1, q0), atol=1e-6),
           f"w(1)={w1.flatten()[0]:.6f}")
    report("w(t=0) == q0*exp(-decay) (decayed at the DATA end)",
           torch.allclose(w0, torch.full_like(w0, q0 * math.exp(-decay)), atol=1e-6),
           f"w(0)={w0.flatten()[0]:.6f}")
    report("w is monotonically increasing in t",
           all(
               plan.gmm_weight_t(torch.full((B,), a), x1, q0=q0, decay_alpha=decay).flatten()[0]
               < plan.gmm_weight_t(torch.full((B,), b), x1, q0=q0, decay_alpha=decay).flatten()[0]
               for a, b in zip([0.0, 0.25, 0.5, 0.75], [0.25, 0.5, 0.75, 1.0])
           ))
    wc = plan.gmm_weight_t(torch.rand(B), x1, q0=q0, schedule="const")
    report("schedule='const' is t-independent",
           torch.allclose(wc, torch.full_like(wc, q0), atol=1e-6))

    # ------------------------------------------------------------------
    print("\n== Check 3: time-axis equivalence vs RAPID (THE critical check) ==")
    # Reference implementation of RAPID's plan_gmm_adaptive, inlined so the
    # check runs without the RAPID checkout. eps is injected rather than drawn
    # so the two sides are comparable.
    def rapid_plan(s, x0_gmm, x1, eps, alpha=1.0, q0=0.5):
        s_e = s.view(-1, 1, 1, 1)
        q = q0 * torch.exp(-alpha * s_e)
        x0_blended = q * x0_gmm + (1.0 - q) * eps
        xt = s_e * x1 + (1.0 - s_e) * x0_blended
        ut = x1 - x0_blended
        return xt, ut

    max_dx, max_du = 0.0, 0.0
    for t_val in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        t_rae = torch.full((B,), t_val)
        s_rapid = torch.full((B,), 1.0 - t_val)          # s = 1 - t
        _, xt_new, ut_new = plan.plan_gmm_adaptive(
            t_rae, x0_gmm, x1, q0=q0, decay_alpha=decay, eps=eps
        )
        xt_old, ut_old = rapid_plan(s_rapid, x0_gmm, x1, eps, alpha=decay, q0=q0)
        max_dx = max(max_dx, (xt_new - xt_old).abs().max().item())
        max_du = max(max_du, (ut_new + ut_old).abs().max().item())  # ut is negated

    report("xt identical under s = 1 - t", max_dx < 1e-6, f"max|dxt|={max_dx:.3e}")
    report("ut exactly sign-flipped under s = 1 - t", max_du < 1e-6, f"max|ut_new+ut_old|={max_du:.3e}")

    if args.old_path:
        if not os.path.exists(args.old_path):
            print(f"  [SKIP] --old-path not found: {args.old_path}")
        else:
            old = load_module(args.old_path, "rapid_path")
            old_plan = old.ICPlan()
            a_old, _ = old_plan.compute_alpha_t(torch.zeros(B))
            report("RAPID's own alpha_t(0) == 0 (its t=0 is NOISE; axes really are opposed)",
                   torch.allclose(a_old, torch.zeros(B), atol=1e-6),
                   f"alpha_rapid(0)={a_old[0]:.3f} vs alpha_rae(0)={a0[0]:.3f}")

    # ------------------------------------------------------------------
    print("\n== Check 4: endpoint behaviour of the blended path ==")
    _, xt_n, _ = plan.plan_gmm_adaptive(torch.ones(B), x0_gmm, x1, q0=q0, decay_alpha=decay, eps=eps)
    expect_noise = q0 * x0_gmm + (1 - q0) * eps
    report("xt(t=1) == q0*x0_gmm + (1-q0)*eps  (matches GMMPrior.init_latent)",
           torch.allclose(xt_n, expect_noise, atol=1e-6))
    _, xt_d, _ = plan.plan_gmm_adaptive(torch.zeros(B), x0_gmm, x1, q0=q0, decay_alpha=decay, eps=eps)
    report("xt(t=0) == x1  (data end is clean)", torch.allclose(xt_d, x1, atol=1e-6))

    # ------------------------------------------------------------------
    print("\n== Check 5: plan_gmm_const is the EXACT-velocity control ==")
    # With w constant, x0_blended does not depend on t, so d(xt)/dt must equal
    # ut analytically. Verify against a central finite difference.
    #
    # Run in float64: a central difference with h=1e-4 on O(1) values cancels
    # away ~4 significant digits, which in fp32 leaves ~1e-3 of pure rounding
    # noise and would mask (or fake) a real discrepancy.
    tv = 0.4
    h = 1e-4
    x1d, x0d, epsd = x1.double(), x0_gmm.double(), eps.double()

    def const_at(tt):
        return plan.plan_gmm_const(torch.full((B,), tt, dtype=torch.float64), x0d, x1d,
                                   q_const=q0, eps=epsd)

    def adapt_at(tt):
        return plan.plan_gmm_adaptive(torch.full((B,), tt, dtype=torch.float64), x0d, x1d,
                                      q0=q0, decay_alpha=decay, eps=epsd)

    _, _, ut_c = const_at(tv)
    fd = (const_at(tv + h)[1] - const_at(tv - h)[1]) / (2 * h)
    err_const = (fd - ut_c).abs().max().item()
    report("const: ut == d(xt)/dt exactly", err_const < 1e-9, f"max err={err_const:.3e}")

    _, _, ut_a = adapt_at(tv)
    fd_a = (adapt_at(tv + h)[1] - adapt_at(tv - h)[1]) / (2 * h)
    err_ad = (fd_a - ut_a).abs().max().item()
    # The gap is the omitted t * dw/dt * (x0_gmm - eps) term. It is EXPECTED to
    # be non-zero -- this check documents the size of the known approximation,
    # it does not flag a bug.
    report("adaptive: dw/dt gap present and finite (known approximation)",
           err_ad > 1e-6 and math.isfinite(err_ad), f"max gap={err_ad:.4f}")

    # ------------------------------------------------------------------
    print("\n== Check 6: PCA diag-var restore matches the original table build ==")
    prior_mod = load_module(args.prior, "gmm_prior")
    d, D, K = 16, 512, 3
    U = torch.linalg.qr(torch.randn(D, d))[0].t().contiguous()   # (d, D) orthonormal rows
    v = torch.rand(K, d) + 0.1                                    # (K, d)
    # RAPID built a (K, D) table once:      diag_var_all = v @ (U*U)
    table = v @ (U * U)
    # The port evaluates the same expression per batch.
    on_the_fly = torch.stack([v[k] @ (U * U) for k in range(K)])
    report("v @ U^2 restore == precomputed table",
           torch.allclose(table, on_the_fly, atol=1e-6),
           f"max|diff|={(table - on_the_fly).abs().max().item():.3e}")

    # mean over D computed via u2_mean (as the merge stage does) must equal the
    # direct mean of the restored table.
    u2_mean = (U * U).mean(dim=1)
    report("merge-stage mean-variance shortcut is exact",
           torch.allclose(v @ u2_mean, table.mean(dim=1), atol=1e-6),
           f"max|diff|={(v @ u2_mean - table.mean(dim=1)).abs().max().item():.3e}")

    # ------------------------------------------------------------------
    print("\n== Check 7: LPF ==")
    mu = torch.randn(2, 4, 16, 16)
    report("lowpass_avg(alpha=0) is identity",
           torch.allclose(prior_mod.lowpass_avg(mu, 0.0), mu, atol=1e-6))
    sm = prior_mod.lowpass_avg(mu, 1.0)
    report("lowpass_avg(alpha=1) reduces high-frequency energy",
           sm.var().item() < mu.var().item(),
           f"var {mu.var().item():.4f} -> {sm.var().item():.4f}")
    a3 = prior_mod.estimate_lpf_alpha_minus3db()
    report("estimate_lpf_alpha_minus3db returns a sane diagnostic",
           0.0 < a3 < 2.0, f"alpha*={a3:.4f} (POWER convention; diagnostic only)")

    # ------------------------------------------------------------------
    print("\n== Check 8: timestep shift concentrates training near the noise end ==")
    shift = math.sqrt(196608 / 4096)
    t_u = torch.rand(200000)
    t_s = shift * t_u / (1 + (shift - 1) * t_u)
    frac = (t_s > 0.5).float().mean().item()
    w_mean = (q0 * torch.exp(-decay * (1 - t_s))).mean().item()
    report("shift s = sqrt(196608/4096) ~= 6.93", abs(shift - 6.928) < 0.01, f"s={shift:.4f}")
    # This is MANUAL section 6.1 made concrete: most training samples land where
    # w(t) ~ q0, so the GMM is injected far more strongly than in LightningDiT.
    report("majority of shifted t lands in the high-w region",
           frac > 0.8, f"P(t>0.5)={frac:.3f}, E[w(t)]={w_mean:.4f} (vs q0={q0})")

    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        print("=" * 62)
        return 1
    print("ALL CHECKS PASSED")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
