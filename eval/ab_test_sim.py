"""
ab_test_sim.py
Simulates an A/B/C test comparing three arms, using held-out tracks as a
proxy for engagement:
  - Treatment:   Full engine   (ALS retrieval + XGBRanker reranking)
  - Als_only:    ALS-only      (ALS retrieval, no reranking -- ablation arm)
  - Control:     Popularity    (most-interacted-with tracks globally)

A pilot sample (PILOT_N playlists, drawn independently of the full sample) is
run through all three arms (same pipeline as the full simulation) to
estimate the paired statistics the power analysis actually needs: the std of
per-playlist differences for continuous metrics, and the discordant-pair
rate for the binary metric (McNemar's test). These are estimated separately
for Treatment-vs-Control and Treatment-vs-ALS-only, since the two
comparisons can have very different variance/discordance.

The number of playlists for the full simulation is then derived from the
power analysis itself (the largest required-n across all metrics and
comparisons), rather than being guessed upfront -- mirroring how a real
experiment's sample size is set. Pass --n-playlists to override this with a
fixed value; if it's below what the power analysis requires, the run
proceeds anyway but is flagged as underpowered.

METRICS:
  Primary:    NDCG@K   -- matches the XGBRanker training objective directly
  Secondary:  Recall@K -- fraction of held-out tracks recovered (continuous)
  Tertiary:   Hit rate -- binary, did at least one held-out track appear

Each metric is analyzed as follows:
  - A parametric paired test (McNemar's test for binary, paired t-test for continuous)
  - A paired bootstrap CI (resamples playlist indices, not each arm's column
    independently, so the pairing is preserved in every replicate)
  - Agreement check between the two (flags if they diverge)

NOTE: This is a simulation, not a live experiment. No real users are randomized.
The held-out split is a proxy for engagement. MDE and power calculations use
the same formulas a real experiment would, and (as of the pilot-driven sizing
above) the sample size decision is made the same way too: from the pilot,
before the full simulation runs.

PAIRED vs UNPAIRED: the primary analysis above is PAIRED -- every playlist is
scored under all three arms against the same held-out split. That's only
possible offline; a real live A/B test must assign each user/playlist to
exactly ONE arm (between-subjects) and compare with independent-sample tests
(Welch's t-test, a two-proportion z-test), not paired ones. Pairing removes
playlist-to-playlist variance from the error term, so for the same true
effect the paired design here is more powerful -- smaller p-values, tighter
CIs, a smaller required-n -- than a live test measuring the identical effect
would be. To make that gap concrete, this script also prints the unpaired
required-n alongside the paired one in the power analysis, and re-splits the
same simulated playlists into three disjoint groups (see `report_unpaired`)
to show what a real, between-subjects live A/B test would actually require
and show.

Run:
    python eval/ab_test_sim.py --k 20 --pilot-n 100
    python eval/ab_test_sim.py --n-playlists 2000  # override auto-sizing
"""

import argparse

import numpy as np
import psycopg
from scipy import stats

from eval.evaluate import (
    DB_URI,
    fetch_playlists,
    fetch_global_popularity,
    split_holdout,
    stable_seed,
    recall_at_k,
    ndcg_at_k,
    get_engine_and_als_recommendations,
    load_models,
)

PILOT_N = 100  # playlists used to estimate baseline before full simulation


# Metric helpers
def binary_hit(recommended, held_out, k):
    """1 if at least one held-out track was recommended, else 0."""
    return 1 if set(recommended[:k]) & set(held_out) else 0


# Statistical tests
def mcnemar_test(treatment_binary, control_binary):
    """
    McNemar's test for paired binary outcomes -- the correct test here since
    every playlist is scored under both arms with the same held-out split.
    Only discordant pairs (where the arms disagree) carry information;
    concordant pairs (both hit or both miss) cancel out and are ignored.

    Returns: b (treatment-only hits), c (control-only hits), chi2 stat, p-value
    """
    treatment_binary = np.asarray(treatment_binary)
    control_binary = np.asarray(control_binary)
    b = int(np.sum((treatment_binary == 1) & (control_binary == 0)))
    c = int(np.sum((treatment_binary == 0) & (control_binary == 1)))
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected
    p_val = stats.chi2.sf(chi2_stat, df=1)
    return b, c, chi2_stat, p_val


def bootstrap_ci(treatment, control, n_boot=5000, ci=95, seed=42):
    """
    Paired bootstrap CI for the difference in means (treatment - control).
    Resamples playlist INDICES (not each arm's column independently) so the
    pairing -- same playlist, same held-out split, under both arms -- is
    preserved in every replicate. Works for binary or continuous metrics.
    """
    rng = np.random.default_rng(seed)
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    n = len(treatment)
    diffs = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (treatment[idx] - control[idx]).mean()

    lower = np.percentile(diffs, (100 - ci) / 2)
    upper = np.percentile(diffs, 100 - (100 - ci) / 2)
    return diffs.mean(), lower, upper


def two_proportion_ztest(binary_a, binary_b):
    """
    Pooled-variance two-proportion z-test for INDEPENDENT binary samples --
    the test a real live A/B test's hit-rate comparison would use, since it
    assigns each user/playlist to exactly one arm rather than scoring every
    playlist under every arm (the analogue to `mcnemar_test`, for unpaired data).

    Returns: p_a, p_b, z stat, p-value
    """
    a = np.asarray(binary_a)
    b = np.asarray(binary_b)
    n_a, n_b = len(a), len(b)
    p_a, p_b = a.mean(), b.mean()
    p_pool = (a.sum() + b.sum()) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return p_a, p_b, 0.0, 1.0
    z_stat = (p_a - p_b) / se
    p_val = 2 * stats.norm.sf(abs(z_stat))
    return p_a, p_b, z_stat, p_val


def bootstrap_ci_unpaired(a, b, n_boot=5000, ci=95, seed=42):
    """
    Independent-groups bootstrap CI for the difference in means. Resamples
    each group SEPARATELY (not shared indices, since the groups aren't
    paired here) -- the analogue to `bootstrap_ci`, for unpaired data.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = np.empty(n_boot)

    for i in range(n_boot):
        a_idx = rng.integers(0, len(a), size=len(a))
        b_idx = rng.integers(0, len(b), size=len(b))
        diffs[i] = a[a_idx].mean() - b[b_idx].mean()

    lower = np.percentile(diffs, (100 - ci) / 2)
    upper = np.percentile(diffs, 100 - (100 - ci) / 2)
    return diffs.mean(), lower, upper


def check_agreement(parametric_excludes_zero, boot_lo, boot_hi, label):
    """
    Checks whether the parametric test and bootstrap CI agree on significance.
    Prints a warning if they diverge -- a signal to inspect the data distribution.
    """
    boot_excludes_zero = (boot_lo > 0) or (boot_hi < 0)
    if parametric_excludes_zero == boot_excludes_zero:
        print(f"  [OK] Parametric test and bootstrap agree on significance ({label})")
    else:
        print(f"  [WARN] Parametric test and bootstrap DISAGREE ({label}) -- inspect data distribution")


# Power analysis
def mde_binary_paired(n, discordant_rate, power=0.8, alpha=0.05):
    """
    Minimum detectable effect for a paired binary metric (McNemar's test).
    Power depends on the discordant-pair rate psi = P(arms disagree on a
    given playlist) -- NOT the marginal hit rate, which is what an
    (incorrect, for this design) independent two-proportion test would use.
    Assumes the target MDE is small relative to psi (standard approximation).
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return (z_alpha + z_beta) * np.sqrt(discordant_rate / n)


def mde_continuous_paired(n, std_diff, power=0.8, alpha=0.05):
    """
    Minimum detectable effect for a paired continuous metric (paired t-test).
    Uses the std of per-playlist differences (treatment - comparison), which
    captures the correlation between arms that pairing is meant to exploit --
    NOT sqrt(2)*std of a single arm, which is the independent two-sample formula.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return (z_alpha + z_beta) * std_diff / np.sqrt(n)


def required_n_binary_paired(discordant_rate, target_mde, power=0.8, alpha=0.05):
    """Inverse of mde_binary_paired."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2 * discordant_rate) / target_mde ** 2
    return int(np.ceil(n))


def required_n_continuous_paired(std_diff, target_mde, power=0.8, alpha=0.05):
    """Inverse of mde_continuous_paired."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) * std_diff / target_mde) ** 2
    return int(np.ceil(n))


def mde_binary_unpaired(n, p_a, p_b, power=0.8, alpha=0.05):
    """
    Minimum detectable effect for two INDEPENDENT proportions (two-proportion
    z-test), each with the same per-arm sample size n -- what a real live A/B
    test's hit-rate comparison needs, since it can't reuse the same playlist
    across arms the way this simulation's paired design does.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_bar = (p_a + p_b) / 2
    return (z_alpha + z_beta) * np.sqrt(2 * p_bar * (1 - p_bar) / n)


def required_n_binary_unpaired(p_a, p_b, target_mde, power=0.8, alpha=0.05):
    """Inverse of mde_binary_unpaired."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_bar = (p_a + p_b) / 2
    n = ((z_alpha + z_beta) ** 2 * 2 * p_bar * (1 - p_bar)) / target_mde ** 2
    return int(np.ceil(n))


def mde_continuous_unpaired(n, std_a, std_b, power=0.8, alpha=0.05):
    """
    Minimum detectable effect for two INDEPENDENT continuous samples (Welch's
    t-test), each with the same per-arm sample size n. Uses each arm's own
    std (NOT the std of per-playlist differences, which relies on the
    pairing this design doesn't have) -- what a real live A/B test needs.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return (z_alpha + z_beta) * np.sqrt((std_a ** 2 + std_b ** 2) / n)


def required_n_continuous_unpaired(std_a, std_b, target_mde, power=0.8, alpha=0.05):
    """Inverse of mde_continuous_unpaired."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2 * (std_a ** 2 + std_b ** 2)) / target_mde ** 2
    return int(np.ceil(n))


def compute_pilot_stats(pilot_results):
    """
    Computes the paired statistics the power analysis needs from a pilot run
    (a `run_simulation`-shaped results dict): the discordant-pair rate for
    the binary metric, and the std of per-playlist differences for the
    continuous metrics. Computed separately for Treatment-vs-Control and
    Treatment-vs-ALS-only, since the two comparisons can differ a lot --
    e.g. Treatment vs Control (a weak baseline) tends to be highly
    discordant/high-variance, while Treatment vs ALS-only (a related model)
    tends to be more correlated.
    """
    t_bin = np.asarray(pilot_results["treatment_binary"])
    a_bin = np.asarray(pilot_results["als_binary"])
    c_bin = np.asarray(pilot_results["control_binary"])
    t_recall = np.asarray(pilot_results["treatment_recall"])
    a_recall = np.asarray(pilot_results["als_recall"])
    c_recall = np.asarray(pilot_results["control_recall"])
    t_ndcg = np.asarray(pilot_results["treatment_ndcg"])
    a_ndcg = np.asarray(pilot_results["als_ndcg"])
    c_ndcg = np.asarray(pilot_results["control_ndcg"])

    return {
        "binary_discordant_rate": {
            "vs_control": np.mean(t_bin != c_bin),
            "vs_als": np.mean(t_bin != a_bin),
        },
        "recall_diff_std": {
            "vs_control": np.std(t_recall - c_recall, ddof=1),
            "vs_als": np.std(t_recall - a_recall, ddof=1),
        },
        "ndcg_diff_std": {
            "vs_control": np.std(t_ndcg - c_ndcg, ddof=1),
            "vs_als": np.std(t_ndcg - a_ndcg, ddof=1),
        },
        # Per-arm (not paired-diff) stats -- feed the unpaired/independent-groups
        # power formulas, which don't get to exploit the pairing correlation.
        "hit_rate": {
            "treatment": np.mean(t_bin),
            "control": np.mean(c_bin),
            "als": np.mean(a_bin),
        },
        "recall_arm_stats": {
            "treatment": (np.mean(t_recall), np.std(t_recall, ddof=1)),
            "control": (np.mean(c_recall), np.std(c_recall, ddof=1)),
            "als": (np.mean(a_recall), np.std(a_recall, ddof=1)),
        },
        "ndcg_arm_stats": {
            "treatment": (np.mean(t_ndcg), np.std(t_ndcg, ddof=1)),
            "control": (np.mean(c_ndcg), np.std(c_ndcg, ddof=1)),
            "als": (np.mean(a_ndcg), np.std(a_ndcg, ddof=1)),
        },
    }


def compute_required_n(pilot_stats, target_mde_hit_rate, target_mde_recall, target_mde_ndcg):
    """
    Computes the sample size required to hit ~80% power for each metric's
    target MDE, for each comparison (vs Control, vs ALS-only), using the
    paired-design pilot stats. Returns the per-metric-per-comparison
    requirements plus the overall requirement (the max across all of them --
    the binding constraint, since the full simulation must be large enough
    to power every reported comparison at once).

    Also computes the UNPAIRED equivalent (binary_unpaired/recall_unpaired/
    ndcg_unpaired/overall_unpaired) from the same pilot's per-arm stats: the
    sample size a real live, between-subjects A/B test would need to detect
    the same target effect, since it can't exploit the playlist-level
    pairing this simulation's design relies on.
    """
    required = {
        "binary": {}, "recall": {}, "ndcg": {},
        "binary_unpaired": {}, "recall_unpaired": {}, "ndcg_unpaired": {},
    }
    for comparison, other_arm in (("vs_control", "control"), ("vs_als", "als")):
        required["binary"][comparison] = required_n_binary_paired(
            pilot_stats["binary_discordant_rate"][comparison], target_mde_hit_rate)
        required["recall"][comparison] = required_n_continuous_paired(
            pilot_stats["recall_diff_std"][comparison], target_mde_recall)
        required["ndcg"][comparison] = required_n_continuous_paired(
            pilot_stats["ndcg_diff_std"][comparison], target_mde_ndcg)

        required["binary_unpaired"][comparison] = required_n_binary_unpaired(
            pilot_stats["hit_rate"]["treatment"], pilot_stats["hit_rate"][other_arm], target_mde_hit_rate)
        _, t_recall_std = pilot_stats["recall_arm_stats"]["treatment"]
        _, other_recall_std = pilot_stats["recall_arm_stats"][other_arm]
        required["recall_unpaired"][comparison] = required_n_continuous_unpaired(
            t_recall_std, other_recall_std, target_mde_recall)
        _, t_ndcg_std = pilot_stats["ndcg_arm_stats"]["treatment"]
        _, other_ndcg_std = pilot_stats["ndcg_arm_stats"][other_arm]
        required["ndcg_unpaired"][comparison] = required_n_continuous_unpaired(
            t_ndcg_std, other_ndcg_std, target_mde_ndcg)

    required["overall"] = max(
        v for metric in ("binary", "recall", "ndcg") for v in required[metric].values()
    )
    required["overall_unpaired"] = max(
        v for metric in ("binary_unpaired", "recall_unpaired", "ndcg_unpaired") for v in required[metric].values()
    )
    return required


def print_power_analysis(pilot_stats, required_n, n_playlists, target_mde_hit_rate, target_mde_recall, target_mde_ndcg):
    """
    Prints MDE for each metric/comparison given the full simulation sample
    size, plus the sample size required to detect the target effect size --
    so the "how many playlists do we need" decision is explicit rather than
    reverse-engineered from whatever n_playlists happened to be passed in.

    Alongside each paired requirement (this simulation's design), also prints
    the UNPAIRED requirement: how many playlists per arm a real live,
    between-subjects A/B test would need for the same target effect, since it
    can't reuse a playlist across arms the way this simulation's paired
    design does. The paired requirement is always <= the unpaired one for the
    same effect size -- that gap is the "power bonus" pairing gets you here,
    which a live experiment won't have.
    """
    print("\n=== Pre-Experiment Power Analysis ===")
    print(f"Full simulation sample size: {n_playlists} playlists per arm (paired design)")
    print(f"(Estimates from independent pilot of {PILOT_N} playlists)\n")

    comparisons = (("vs_control", "vs Control"), ("vs_als", "vs ALS-only"))

    def _report_paired(label, mde_fn, stat_key, target_mde, req_n_by_comparison, unit_desc):
        print(f"{label} [paired -- this simulation's design]:")
        for key, comp_label in comparisons:
            stat_val = pilot_stats[stat_key][key]
            mde_at_n = mde_fn(n_playlists, stat_val)
            req_n = req_n_by_comparison[key]
            print(f"  {comp_label} -- MDE at n={n_playlists}: {mde_at_n:.4f}")
            print(f"    Target effect size: {target_mde:.4f} {unit_desc} -- requires n >= {req_n} playlists/arm")
            if n_playlists >= req_n:
                print(f"    [OK] n={n_playlists} meets or exceeds the required sample size ({req_n}).")
            else:
                print(f"    [WARN] n={n_playlists} is BELOW the required sample size ({req_n}) "
                      f"to detect a {target_mde:.4f} lift with ~80% power.")

    def _report_unpaired(label, req_n_by_comparison):
        print(f"{label} [unpaired -- what a real live A/B test needs]:")
        for key, comp_label in comparisons:
            print(f"  {comp_label} -- requires n >= {req_n_by_comparison[key]} playlists/arm (independent groups)")

    _report_paired("Binary hit rate", mde_binary_paired, "binary_discordant_rate",
                   target_mde_hit_rate, required_n["binary"], "absolute lift")
    _report_unpaired("Binary hit rate", required_n["binary_unpaired"])
    print()
    _report_paired("Continuous recall", mde_continuous_paired, "recall_diff_std",
                   target_mde_recall, required_n["recall"], "mean difference")
    _report_unpaired("Continuous recall", required_n["recall_unpaired"])
    print()
    _report_paired("NDCG@K", mde_continuous_paired, "ndcg_diff_std",
                   target_mde_ndcg, required_n["ndcg"], "mean NDCG difference")
    _report_unpaired("NDCG@K", required_n["ndcg_unpaired"])

    print(f"\nOverall required n, paired (this simulation's design): {required_n['overall']} playlists/arm")
    print(f"Overall required n, unpaired (a real live/randomized test): {required_n['overall_unpaired']} playlists/arm")
    if n_playlists == required_n["overall"]:
        print("Full simulation sample size was set from the paired power analysis above.\n")
    else:
        print(f"NOTE: --n-playlists override was used ({n_playlists}) instead of the "
              f"paired-power-analysis-derived value ({required_n['overall']}).\n")


# Simulation
def run_simulation(playlists, als_model, user_item_matrix, xgb_model, popularity_baseline, k=20, holdout_frac=0.2, seed=None, conn=None):
    """
    Runs all three arms over the full playlist sample. Each playlist sees the
    engine (treatment), ALS-only (ablation arm), and popularity baseline
    (control) against the same held-out split.
    """
    treatment_binary, als_binary, control_binary = [], [], []
    treatment_recall, als_recall, control_recall = [], [], []
    treatment_ndcg, als_ndcg, control_ndcg = [], [], []

    for index, (pid, track_ids) in enumerate(playlists.items()):
        print(f"Simulating playlist {index + 1} of {len(playlists)}")
        seed_tracks, held_out = split_holdout(track_ids, holdout_frac, seed=stable_seed(pid, seed))

        recs_treatment, recs_als = get_engine_and_als_recommendations(
            seed_tracks, als_model, user_item_matrix, xgb_model, k=k, conn=conn)
        recs_control = popularity_baseline

        treatment_binary.append(binary_hit(recs_treatment, held_out, k))
        als_binary.append(binary_hit(recs_als, held_out, k))
        control_binary.append(binary_hit(recs_control, held_out, k))

        treatment_recall.append(recall_at_k(recs_treatment, held_out, k))
        als_recall.append(recall_at_k(recs_als, held_out, k))
        control_recall.append(recall_at_k(recs_control, held_out, k))

        treatment_ndcg.append(ndcg_at_k(recs_treatment, held_out, k))
        als_ndcg.append(ndcg_at_k(recs_als, held_out, k))
        control_ndcg.append(ndcg_at_k(recs_control, held_out, k))

    return {
        "treatment_binary": treatment_binary, "als_binary": als_binary, "control_binary": control_binary,
        "treatment_recall": treatment_recall, "als_recall": als_recall, "control_recall": control_recall,
        "treatment_ndcg": treatment_ndcg, "als_ndcg": als_ndcg, "control_ndcg": control_ndcg,
    }


# Reporting
def _report_metric(label, k_desc, treatment, arm_b, control, test_fn, test_label_a, test_label_b):
    """Reports Treatment vs Control and Treatment vs ALS-only for one metric."""
    print(f"\n{label} {k_desc}:")

    t_stat, p_val = test_fn(treatment, control)
    print(f"  Treatment: {np.mean(treatment):.4f}   Control: {np.mean(control):.4f}   "
          f"Lift: {np.mean(treatment) - np.mean(control):+.4f}")
    print(f"  {test_label_a}: t = {t_stat:.3f}, p = {p_val:.4f}")
    mean_diff, lo, hi = bootstrap_ci(treatment, control)
    print(f"  Bootstrap 95% CI vs Control: [{lo:.4f}, {hi:.4f}]  (mean diff: {mean_diff:.4f})")
    check_agreement(p_val < 0.05, lo, hi, f"{label} vs Control")

    t_stat_als, p_val_als = test_fn(treatment, arm_b)
    print(f"  ALS-only:  {np.mean(arm_b):.4f}   Lift over ALS-only: {np.mean(treatment) - np.mean(arm_b):+.4f}")
    print(f"  {test_label_b}: t = {t_stat_als:.3f}, p = {p_val_als:.4f}")
    mean_diff_als, lo_als, hi_als = bootstrap_ci(treatment, arm_b)
    print(f"  Bootstrap 95% CI vs ALS-only: [{lo_als:.4f}, {hi_als:.4f}]  (mean diff: {mean_diff_als:.4f})")
    check_agreement(p_val_als < 0.05, lo_als, hi_als, f"{label} vs ALS-only")


def _report_binary_vs(treatment_binary, other_binary, other_name):
    """Reports Treatment vs `other_name` for the binary hit-rate metric via McNemar's test."""
    p_t = np.mean(treatment_binary)
    p_o = np.mean(other_binary)
    b, c, chi2_stat, p_val = mcnemar_test(treatment_binary, other_binary)
    print(f"  {other_name}: {p_o:.4f}   Lift over {other_name}: {p_t - p_o:+.4f}")
    print(f"  McNemar's test vs {other_name}: b={b} (treatment-only hits), c={c} ({other_name}-only hits), "
          f"chi2 = {chi2_stat:.3f}, p = {p_val:.4f}")
    mean_diff, lo, hi = bootstrap_ci(treatment_binary, other_binary)
    print(f"  Bootstrap 95% CI vs {other_name}: [{lo:.4f}, {hi:.4f}]  (mean diff: {mean_diff:.4f})")
    check_agreement(p_val < 0.05, lo, hi, f"Hit rate vs {other_name}")


def _report_metric_unpaired(label, treatment, arm_b, control, arm_b_name):
    """Reports Treatment vs Control and Treatment vs `arm_b_name` for one continuous
    metric as INDEPENDENT groups (Welch's t-test) -- the unpaired analogue to
    `_report_metric`."""
    print(f"\n{label} (unpaired, independent groups):")

    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
    print(f"  Treatment: {np.mean(treatment):.4f} (n={len(treatment)})   "
          f"Control: {np.mean(control):.4f} (n={len(control)})   "
          f"Lift: {np.mean(treatment) - np.mean(control):+.4f}")
    print(f"  Welch's t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
    mean_diff, lo, hi = bootstrap_ci_unpaired(treatment, control)
    print(f"  Bootstrap 95% CI vs Control: [{lo:.4f}, {hi:.4f}]  (mean diff: {mean_diff:.4f})")
    check_agreement(p_val < 0.05, lo, hi, f"{label} vs Control (unpaired)")

    t_stat_b, p_val_b = stats.ttest_ind(treatment, arm_b, equal_var=False)
    print(f"  {arm_b_name}: {np.mean(arm_b):.4f} (n={len(arm_b)})   "
          f"Lift over {arm_b_name}: {np.mean(treatment) - np.mean(arm_b):+.4f}")
    print(f"  Welch's t-test: t = {t_stat_b:.3f}, p = {p_val_b:.4f}")
    mean_diff_b, lo_b, hi_b = bootstrap_ci_unpaired(treatment, arm_b)
    print(f"  Bootstrap 95% CI vs {arm_b_name}: [{lo_b:.4f}, {hi_b:.4f}]  (mean diff: {mean_diff_b:.4f})")
    check_agreement(p_val_b < 0.05, lo_b, hi_b, f"{label} vs {arm_b_name} (unpaired)")


def _report_binary_vs_unpaired(treatment_binary, other_binary, other_name):
    """Reports Treatment vs `other_name` for the binary hit-rate metric as
    INDEPENDENT groups via `two_proportion_ztest` -- the unpaired analogue to
    `_report_binary_vs`."""
    p_t, p_o, z_stat, p_val = two_proportion_ztest(treatment_binary, other_binary)
    print(f"  {other_name}: {p_o:.4f} (n={len(other_binary)})   Lift over {other_name}: {p_t - p_o:+.4f}")
    print(f"  Two-proportion z-test vs {other_name}: z = {z_stat:.3f}, p = {p_val:.4f}")
    mean_diff, lo, hi = bootstrap_ci_unpaired(treatment_binary, other_binary)
    print(f"  Bootstrap 95% CI vs {other_name}: [{lo:.4f}, {hi:.4f}]  (mean diff: {mean_diff:.4f})")
    check_agreement(p_val < 0.05, lo, hi, f"Hit rate vs {other_name} (unpaired)")


def report_unpaired(results, k, required_n_unpaired=None):
    """
    Re-analyzes the SAME simulation results as an independent-groups
    (unpaired) comparison -- what a real live, between-subjects A/B test
    would actually show, since it must assign each playlist to exactly one
    arm rather than scoring every playlist under all three the way the
    paired `report()` above does.

    Doesn't re-run the simulation: splits the same n playlists into three
    disjoint groups by index % 3 (the sample order is already randomized via
    the seeded SQL fetch, so this is an effectively-random 3-way split) and
    keeps only ONE arm's value per playlist -- group 0's treatment values,
    group 1's control values, group 2's als-only values. No playlist
    contributes to more than one arm's group here, mirroring the constraint
    a live test faces, at zero extra DB/model cost. Each arm's group is
    therefore ~n/3 rather than the full n.
    """
    n = len(results["treatment_ndcg"])
    group = np.arange(n) % 3
    treat_idx, control_idx, als_idx = group == 0, group == 1, group == 2

    def _split(key, idx):
        return np.asarray(results[key])[idx]

    print("\n=== Unpaired (Independent-Groups) Equivalent: what a real live A/B test would show ===\n")
    print("A real live experiment can't show the same playlist two arms, so it must split playlists")
    print("into disjoint groups -- one arm per group -- and compare with independent-sample tests,")
    print("not the paired tests above. This re-splits the same playlists into three disjoint groups")
    print(f"({int(treat_idx.sum())}/{int(control_idx.sum())}/{int(als_idx.sum())} playlists) so no")
    print("playlist is scored under more than one arm, then re-analyzes with Welch's t-test / a")
    print("two-proportion z-test.")

    print(f"\n[PRIMARY] NDCG@{k}:")
    _report_metric_unpaired(
        "NDCG",
        _split("treatment_ndcg", treat_idx), _split("als_ndcg", als_idx), _split("control_ndcg", control_idx),
        "ALS-only",
    )

    print(f"\n[SECONDARY] Recall@{k}:")
    _report_metric_unpaired(
        "Recall",
        _split("treatment_recall", treat_idx), _split("als_recall", als_idx), _split("control_recall", control_idx),
        "ALS-only",
    )

    print(f"\n[TERTIARY] Binary hit rate (>=1 relevant track in Top {k}):")
    t_bin = _split("treatment_binary", treat_idx)
    print(f"  Treatment: {np.mean(t_bin):.4f} (n={len(t_bin)})")
    _report_binary_vs_unpaired(t_bin, _split("control_binary", control_idx), "Control")
    _report_binary_vs_unpaired(t_bin, _split("als_binary", als_idx), "ALS-only")

    if required_n_unpaired is not None:
        group_n = int(treat_idx.sum())
        print(f"\nEach arm's group here has ~{group_n} playlists; the pre-experiment unpaired power "
              f"analysis said a live test needs >= {required_n_unpaired} per arm for the target MDE.")
        if group_n < required_n_unpaired:
            print("[WARN] This unpaired split is underpowered relative to that requirement.")


def report(results, n_playlists, k):
    print("\n=== A/B/C Test Simulation Results: Full Engine vs ALS-only vs Popularity ===\n")
    print("(All comparisons are paired -- the same playlists are scored under every arm --")
    print(" so paired tests (t-test, McNemar's) are used rather than independent-sample tests.)")

    print(f"\n[PRIMARY] NDCG@{k} (matches XGBRanker training objective):")
    _report_metric(
        "NDCG", "",
        results["treatment_ndcg"], results["als_ndcg"], results["control_ndcg"],
        stats.ttest_rel, "paired t-test", "paired t-test",
    )

    print(f"\n[SECONDARY] Recall@{k} (fraction of held-out tracks recovered):")
    _report_metric(
        "Recall", "",
        results["treatment_recall"], results["als_recall"], results["control_recall"],
        stats.ttest_rel, "paired t-test", "paired t-test",
    )

    print(f"\n[TERTIARY] Binary hit rate (>=1 relevant track in Top {k}):")
    t_bin, a_bin, c_bin = results["treatment_binary"], results["als_binary"], results["control_binary"]
    print(f"  Treatment: {np.mean(t_bin):.4f}")
    _report_binary_vs(t_bin, c_bin, "Control")
    _report_binary_vs(t_bin, a_bin, "ALS-only")

    print(f"\nN playlists simulated: {n_playlists}")


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-playlists", type=int, default=None,
                         help="Override the full simulation's sample size instead of deriving it "
                              "from the pilot's power analysis. If below the computed requirement, "
                              "the run proceeds but is flagged as underpowered.")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--pilot-n", type=int, default=PILOT_N)
    parser.add_argument("--target-mde-hit-rate", type=float, default=0.03,
                         help="Minimum absolute lift in hit rate worth detecting (default: 0.03 = 3pp)")
    parser.add_argument("--target-mde-recall", type=float, default=0.03,
                         help="Minimum mean difference in Recall@K worth detecting (default: 0.03)")
    parser.add_argument("--target-mde-ndcg", type=float, default=0.03,
                         help="Minimum mean difference in NDCG@K worth detecting (default: 0.03)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed for playlist sampling and holdout splits, for reproducible "
                              "runs. The pilot and full sample use seed and seed+1 respectively, "
                              "so they stay independent even though both are deterministic.")
    args = parser.parse_args()

    als_model, user_item_matrix, xgb_model = load_models()

    conn = psycopg.connect(DB_URI)

    popularity_baseline = fetch_global_popularity(conn, top_n=args.k)

    # Step 1: pilot (drawn independently of the full sample, not a subset of it --
    # a real experiment wouldn't reuse pilot traffic in the final analysis). Runs
    # all three arms through the same pipeline as the full simulation, since the
    # paired power analysis needs per-playlist differences/discordance, not just
    # a single arm's marginal rate.
    print(f"Running pilot on {args.pilot_n} playlists to estimate paired statistics...")
    pilot_playlists = fetch_playlists(conn, args.pilot_n, seed=args.seed)
    pilot_results = run_simulation(
        pilot_playlists, als_model, user_item_matrix, xgb_model,
        popularity_baseline, k=args.k, holdout_frac=args.holdout_frac, seed=args.seed, conn=conn,
    )
    pilot_stats = compute_pilot_stats(pilot_results)

    # Step 2: power analysis -- the full sample size is derived from this,
    # not guessed upfront, unless --n-playlists explicitly overrides it.
    required_n = compute_required_n(
        pilot_stats, args.target_mde_hit_rate, args.target_mde_recall, args.target_mde_ndcg,
    )
    n_playlists = args.n_playlists if args.n_playlists is not None else required_n["overall"]
    print_power_analysis(
        pilot_stats, required_n, n_playlists,
        args.target_mde_hit_rate, args.target_mde_recall, args.target_mde_ndcg,
    )

    # Step 3: full simulation, sized by the power analysis above
    print("=== Running Full Simulation ===\n")
    all_playlists = fetch_playlists(conn, n_playlists, seed=args.seed + 1)
    results = run_simulation(
        all_playlists, als_model, user_item_matrix, xgb_model,
        popularity_baseline, k=args.k, holdout_frac=args.holdout_frac, seed=args.seed + 1, conn=conn,
    )

    # Step 4: results -- paired (this simulation's design), then the unpaired
    # equivalent (what a real live A/B test would show/need)
    report(results, len(all_playlists), args.k)
    report_unpaired(results, args.k, required_n_unpaired=required_n["overall_unpaired"])

    conn.close()
