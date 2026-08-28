"""Recovery-Reversal V1 core: historical music-contrast reconstruction + CPU sensitivity.

STATUS: DRAFT support code for RECOVERY-REVERSAL-V1 (docs/recovery_reversal_v1.md).
NO AudioCaps outcome data is read or produced here. GPU is never touched.

Two independent, importable pieces (numpy only, so both frozen venvs can run them):

1. RECONSTRUCTION (`reconstruct_music_grids`, `r_music`) — rebuild the frozen 64 prompts
   x 3 paired replicate music contrast from the persisted phenomenon artifacts
   (`_phenom_groups_in.json` + `_phenom_groups_out.json`) WITHOUT rescoring any WAV.
   The per-(prompt,replicate) paired difference is
       d[p,r] = C_recovered_off[p,r] - C_pruned_ema_reconstructed_off[p,r]
   (paired by generation seed r across backbones — derive_paired_seed(salt, ytid, r)).
   R_music = mean_p ( mean_r d[p,r] ). The regression target (ledger RECOVERY-REVERSAL-
   AUDIT-1, frozen seed 20260826) is R_music = -0.0941, CI95 [-0.1241, -0.0646].

2. SENSITIVITY (`decompose_variance`, `simulate_design`) — a random-effects preflight for
   the *future* AudioCaps arm (96 prompts x 2 paired replicates). Variance components are
   the plug-in estimate from the historical music paired differences; the AudioCaps effect
   R_AC is a hypothesised true value, never observed data. All randomness derives from
   PCG64(20260827) via SeedSequence spawning, so results are order-invariant and exactly
   reproducible.

The bootstrap CI is the ONE frozen definition (research_pruning.eval.cluster_bootstrap:
prompt-cluster percentile, B=10000). Historical reconstruction uses the historical seed
20260826; the prospective V1 design uses BOOTSTRAP_SEED_V1 = 20260827.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from research_pruning.eval.cluster_bootstrap import cluster_percentile_ci

# --- prospective V1 constants (later decision; NOT the historical 20260826) ----------
BOOTSTRAP_SEED_V1 = 20260827
SELECTION_SALT_V1 = "RECOVERY-REVERSAL-V1|AUDIOCAPS-TEST|2026-08-27"
SESOI = 0.025                     # practical minimum on the R_AC POINT estimate
N_PROMPTS_V1 = 96
N_REPLICATES_V1 = 2
N_PROMPTS_MUSIC = 64
N_REPLICATES_MUSIC = 3
# regression target for the frozen historical music contrast (ledger, seed 20260826)
R_MUSIC_TARGET_POINT = -0.0941
R_MUSIC_TARGET_LO = -0.1241
R_MUSIC_TARGET_HI = -0.0646

RECOVERED_OFF = "p1_recovered__off"
PRUNED_OFF = "p1_pruned_ema_reconstructed__off"
_WAV_RE = re.compile(r"_p(\d+)_r(\d+)\.wav$")


def parse_prompt_replicate(wav_path: str) -> tuple[int, int]:
    """Extract (prompt_index, replicate_index) from a `..._p{P}_r{R}.wav` basename."""
    m = _WAV_RE.search(wav_path)
    if not m:
        raise ValueError(f"cannot parse prompt/replicate from wav path: {wav_path!r}")
    return int(m.group(1)), int(m.group(2))


def _group_grid(items: list, cosines: list, n_prompts: int, n_reps: int):
    """Assemble a (n_prompts, n_reps) score grid + the parsed (p,r) order list.

    Fails loudly on: length mismatch, non-canonical ordering (the frozen scorer scored
    items in list order = (prompt, replicate) ascending), duplicate cells, missing cells.
    """
    if len(items) != n_prompts * n_reps or len(cosines) != n_prompts * n_reps:
        raise ValueError(f"expected {n_prompts * n_reps} items/cosines, "
                         f"got {len(items)}/{len(cosines)}")
    order = [parse_prompt_replicate(it["wav"]) for it in items]
    canonical = [(p, r) for p in range(n_prompts) for r in range(n_reps)]
    if order != canonical:
        raise ValueError("phenom group order is NOT canonical (prompt, replicate) ascending; "
                         "reconstruction pairing would be wrong")
    grid = np.full((n_prompts, n_reps), np.nan, dtype=np.float64)
    captions = [None] * n_prompts
    for (p, r), it, c in zip(order, items, cosines):
        grid[p, r] = c
        captions[p] = it["caption"]
    if np.isnan(grid).any():
        raise ValueError("score grid has missing cells")
    return grid, captions


@dataclass(frozen=True)
class MusicReconstruction:
    recovered: np.ndarray          # (64, 3) C_recovered_off
    pruned: np.ndarray             # (64, 3) C_pruned_ema_reconstructed_off
    paired_diff: np.ndarray        # (64, 3) recovered - pruned (paired by seed)
    prompt_mean_diff: np.ndarray   # (64,) mean over replicates
    captions: list                 # 64 captions (order = prompt_index)
    ytids: list = field(default=None)  # 64 ytids, filled by caller from the battery


def reconstruct_music_grids(groups_in: dict, groups_out: dict,
                            n_prompts: int = N_PROMPTS_MUSIC,
                            n_reps: int = N_REPLICATES_MUSIC) -> MusicReconstruction:
    """Rebuild the paired 64x3 music grids from the two persisted phenom artifacts.

    `groups_in`  = json of `_phenom_groups_in.json`  ({'groups': [{name, items:[{caption,wav}]}]})
    `groups_out` = json of `_phenom_groups_out.json` ({'results':[{name, n, cosines}]})
    The OFF (no-adapter) standalone arms are the ones used for the cross-domain contrast.
    """
    gin = {g["name"]: g["items"] for g in groups_in["groups"]}
    gout = {g["name"]: g["cosines"] for g in groups_out["results"]}
    for name in (RECOVERED_OFF, PRUNED_OFF):
        if name not in gin or name not in gout:
            raise KeyError(f"missing group {name!r} in phenom artifacts")
    rec, cap_r = _group_grid(gin[RECOVERED_OFF], gout[RECOVERED_OFF], n_prompts, n_reps)
    pru, cap_p = _group_grid(gin[PRUNED_OFF], gout[PRUNED_OFF], n_prompts, n_reps)
    if cap_r != cap_p:
        raise ValueError("recovered/pruned OFF caption order disagree; pairing unsafe")
    diff = rec - pru
    return MusicReconstruction(recovered=rec, pruned=pru, paired_diff=diff,
                               prompt_mean_diff=diff.mean(axis=1), captions=cap_r)


def r_music(recon: MusicReconstruction, seed: int = 20260826):
    """Frozen historical R_music CI (prompt-cluster percentile bootstrap, historical seed).

    Returns the cluster_bootstrap.CI over the 64 per-prompt paired-mean differences.
    """
    return cluster_percentile_ci(recon.prompt_mean_diff, seed=seed)


# --- sensitivity: random-effects variance decomposition + design simulation -----------

@dataclass(frozen=True)
class VarComponents:
    grand_mean: float      # mean of the paired difference (== R_music point)
    sigma2_between: float  # prompt random-effect variance of the paired diff
    sigma2_within: float   # within-prompt / per-replicate variance of the paired diff
    n_prompts: int
    n_reps: int

    def as_dict(self) -> dict:
        return {"grand_mean": self.grand_mean, "sigma2_between": self.sigma2_between,
                "sigma2_within": self.sigma2_within, "sigma_between": self.sigma2_between ** 0.5,
                "sigma_within": self.sigma2_within ** 0.5,
                "n_prompts": self.n_prompts, "n_reps": self.n_reps}


def decompose_variance(paired_diff: np.ndarray) -> VarComponents:
    """One-way random-effects decomposition of the paired difference d[p,r].

        d[p,r] = mu + a_p + e[p,r],   a_p ~ (0, s2_between),  e[p,r] ~ (0, s2_within)

    Balanced ANOVA moment estimator (k = reps per prompt):
        MSW = SS_within / (N (k-1));  s2_within = MSW
        MSB = k * SS_between / (N-1); s2_between = max(0, (MSB - MSW) / k)
    s2_within is the PER-REPLICATE variance, so projecting to a different replicate count
    k' uses s2_within / k' (this is exactly how the 3-rep music noise maps onto 2-rep AC).
    """
    d = np.asarray(paired_diff, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"expected (n_prompts, n_reps), got {d.shape}")
    n, k = d.shape
    if k < 2:
        raise ValueError("need >= 2 replicates per prompt to separate within variance")
    grand = float(d.mean())
    prompt_means = d.mean(axis=1)
    ss_within = float(((d - prompt_means[:, None]) ** 2).sum())
    ss_between = float((k * (prompt_means - grand) ** 2).sum())
    msw = ss_within / (n * (k - 1))
    msb = ss_between / (n - 1)
    s2_within = msw
    s2_between = max(0.0, (msb - msw) / k)
    return VarComponents(grand_mean=grand, sigma2_between=s2_between,
                         sigma2_within=s2_within, n_prompts=n, n_reps=k)


def analytic_halfwidth(vc: VarComponents, n_prompts: int, n_reps: int,
                       between_scale: float = 1.0, within_scale: float = 1.0,
                       z: float = 1.959963984540054) -> float:
    """Normal-approx 95% CI half-width of the design mean, cross-check for the bootstrap.

    Var(mean) = (s2_between*bs + (s2_within*ws)/n_reps) / n_prompts.
    """
    var_prompt = vc.sigma2_between * between_scale + (vc.sigma2_within * within_scale) / n_reps
    return z * (var_prompt / n_prompts) ** 0.5


def simulate_design(vc: VarComponents, music_prompt_diff: np.ndarray, *,
                    r_ac: float, n_prompts: int = N_PROMPTS_V1, n_reps: int = N_REPLICATES_V1,
                    between_scale: float = 1.0, within_scale: float = 1.0,
                    n_sim: int = 2000, b_boot: int = 2000, sesoi: float = SESOI,
                    seed_seq: np.random.SeedSequence | None = None) -> dict:
    """Monte-Carlo preflight for one (R_AC effect, variance scenario) cell.

    For each of `n_sim` simulated AudioCaps datasets (n_prompts x n_reps, paired):
        a_p  ~ N(0, s2_between * between_scale)
        e_pr ~ N(0, s2_within  * within_scale)
        d[p,r] = r_ac + a_p + e_pr ;  prompt scalar = mean_r d[p,r]
    then the frozen prompt-cluster percentile bootstrap (B=b_boot) gives (point, lo, hi) of
    R_AC. The interaction I = R_AC - R_music is evaluated with the HISTORICAL music
    uncertainty retained: a joint two-sample bootstrap resamples the 96 AC prompt scalars
    and, independently, the 64 historical music per-prompt diffs, I* = mean(AC*) - mean(music*).

    Reports, over the n_sim datasets:
      * point mean / sd  (expected point-estimate variability)
      * mean CI half-width
      * P(lower_CI95(R_AC) > 0)
      * P(point >= SESOI AND lower_CI95(R_AC) > 0)   [the full R_AC requirement]
      * P(lower_CI95(I) > 0)
      * P(all three PASS conditions)                 [R_AC point>=SESOI, lo(R_AC)>0, lo(I)>0]
    """
    if seed_seq is None:
        seed_seq = np.random.SeedSequence(BOOTSTRAP_SEED_V1)
    rng = np.random.default_rng(seed_seq)
    sb = (vc.sigma2_between * between_scale) ** 0.5
    sw = (vc.sigma2_within * within_scale) ** 0.5
    music = np.asarray(music_prompt_diff, dtype=np.float64)
    n_music = music.size

    points = np.empty(n_sim); halfwidths = np.empty(n_sim)
    lo_pos = np.empty(n_sim, bool); rac_ok = np.empty(n_sim, bool)
    i_lo_pos = np.empty(n_sim, bool); pass_all = np.empty(n_sim, bool)

    for s in range(n_sim):
        a = rng.normal(0.0, sb, size=n_prompts) if sb > 0 else np.zeros(n_prompts)
        e = rng.normal(0.0, sw, size=(n_prompts, n_reps))
        d = r_ac + a[:, None] + e
        prompt_scalar = d.mean(axis=1)                 # (n_prompts,)
        # R_AC bootstrap (prompt cluster)
        idx = rng.integers(0, n_prompts, size=(b_boot, n_prompts))
        boot = prompt_scalar[idx].mean(axis=1)
        point = float(prompt_scalar.mean())
        lo = float(np.percentile(boot, 2.5)); hi = float(np.percentile(boot, 97.5))
        # I = R_AC - R_music, joint two-sample bootstrap (music uncertainty retained)
        midx = rng.integers(0, n_music, size=(b_boot, n_music))
        i_boot = boot - music[midx].mean(axis=1)
        i_lo = float(np.percentile(i_boot, 2.5))

        points[s] = point; halfwidths[s] = (hi - lo) / 2.0
        lo_pos[s] = lo > 0.0
        rac_ok[s] = (point >= sesoi) and (lo > 0.0)
        i_lo_pos[s] = i_lo > 0.0
        pass_all[s] = (point >= sesoi) and (lo > 0.0) and (i_lo > 0.0)

    return {
        "r_ac_true": r_ac, "n_prompts": n_prompts, "n_reps": n_reps,
        "between_scale": between_scale, "within_scale": within_scale,
        "n_sim": n_sim, "b_boot": b_boot, "sesoi": sesoi,
        "point_mean": float(points.mean()), "point_sd": float(points.std(ddof=1)),
        "mean_ci_halfwidth": float(halfwidths.mean()),
        "analytic_ci_halfwidth": analytic_halfwidth(vc, n_prompts, n_reps, between_scale, within_scale),
        "P_lowerCI_Rac_gt0": float(lo_pos.mean()),
        "P_Rac_requirement": float(rac_ok.mean()),
        "P_lowerCI_I_gt0": float(i_lo_pos.mean()),
        "P_pass_all_three": float(pass_all.mean()),
    }
