"""
ab_test_sim.py
Simulates a live A/B/C test comparing three arms, using held-out tracks as a
proxy for engagement:
  - Treatment:   Full engine   (ALS retrieval + XGBRanker reranking)
  - Als_only:    ALS-only      (ALS retrieval, no reranking -- ablation arm)
  - Control:     Popularity    (most-interacted-with tracks globally)

This is the LIVE-EXPERIMENT counterpart to evaluate.py. evaluate.py scores
every playlist under every arm and uses PAIRED tests -- a within-subjects
offline comparison that squeezes maximum power out of a fixed set of
playlists. A real live A/B test can't do that: it must randomize each
user/playlist to EXACTLY ONE arm (between-subjects) and compare arms with
independent-sample tests. This script simulates that design:

  - Each playlist is assigned to exactly one arm (a 3-way random split), so no
    playlist is ever scored under more than one arm.
  - Comparisons use independent-sample tests: Welch's t-test for the
    continuous metrics (NDCG@K, Recall@K), a two-proportion z-test for the
    binary hit-rate metric.
  - The bootstrap CI resamples each arm's group independently (not shared
    indices), matching the unpaired design.

A side benefit of one-arm-per-playlist assignment: only ~1/3 of playlists run
the expensive full-engine pipeline (treatment), ~1/3 run ALS-only retrieval,
and ~1/3 are effectively free (popularity is precomputed) -- so the full
simulation is much cheaper than scoring every playlist under every arm.

A pilot sample (PILOT_N playlists, drawn independently of the full sample) is
scored under ALL three arms -- the pilot is small, so scoring every arm on
every pilot playlist is cheap and gives the most precise per-arm mean/variance
estimates the power analysis needs. From those, the required PER-ARM sample
size is derived (the largest across all metrics/comparisons), and the full
simulation is sized to ~3x that in total (one arm per playlist, so three arms
need 3x the per-arm n). Pass --n-playlists to override the TOTAL sample size;
if it implies fewer than the required per-arm n, the run proceeds but is
flagged as underpowered.

METRICS:
  Primary:    NDCG@K   -- matches the XGBRanker training objective directly
  Secondary:  Recall@K -- fraction of held-out tracks recovered (continuous)
  Tertiary:   Hit rate -- binary, did at least one held-out track appear

NOTE: This is a simulation, not a live experiment. No real users are
randomized; the held-out split is a proxy for engagement. But the design and
statistics mirror a real between-subjects A/B test: pilot-driven sample
sizing, one-arm-per-playlist assignment, and independent-sample tests.

Run:
    python eval/ab_test_sim.py --k 20 --pilot-n 100
    python eval/ab_test_sim.py --n-playlists 3000  # override auto-sizing (total across arms)
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
from src.recommend import get_als_candidates, rank_candidates, get_als_only_recommendations

PILOT_N = 100  # playlists used to estimate per-arm stats before the full simulation

# index % 3 -> arm, for the one-arm-per-playlist assignment in the full simulation.
# The fetch is already seed-shuffled, so this is an effectively-random 3-way split.
ARMS = ("treatment", "control", "als")


# Metric helpers
def binary_hit(recommended, held_out, k):
    """1 if at least one held-out track was recommended, else 0."""
    return 1 if set(recommended[:k]) & set(held_out) else 0


# Statistical tests
def two_proportion_ztest(binary_a, binary_b):
    """
    Pooled-variance two-proportion z-test for INDEPENDENT binary samples --
    the test a real live A/B test's hit-rate comparison uses, since it assigns
    each user/playlist to exactly one arm rather than scoring every playlist
    under every arm.

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
    Independent-groups bootstrap CI for the difference in means. Resamples each
    group SEPARATELY (not shared indices, since the groups aren't paired here).
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


# Power analysis (independent-groups / unpaired)
def mde_binary_unpaired(n, p_a, p_b, power=0.8, alpha=0.05):
    """
    Minimum detectable effect for two INDEPENDENT proportions (two-proportion
    z-test), each with per-arm sample size n -- what a real live A/B test's
    hit-rate comparison needs, since it can't reuse a playlist across arms.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_bar = (p_a + p_b) / 2
    return (z_alpha + z_beta) * np.sqrt(2 * p_bar * (1 - p_bar) / n)


def required_n_binary_unpaired(p_a, p_b, target_mde, power=0.8, alpha=0.05):
    """Inverse of mde_binary_unpaired. Returns required PER-ARM sample size."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_bar = (p_a + p_b) / 2
    n = ((z_alpha + z_beta) ** 2 * 2 * p_bar * (1 - p_bar)) / target_mde ** 2
    return int(np.ceil(n))


def mde_continuous_unpaired(n, std_a, std_b, power=0.8, alpha=0.05):
    """
    Minimum detectable effect for two INDEPENDENT continuous samples (Welch's
    t-test), each with per-arm sample size n. Uses each arm's own std (NOT the
    std of per-playlist differences, which relies on pairing this design
    doesn't have) -- what a real live A/B test needs.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return (z_alpha + z_beta) * np.sqrt((std_a ** 2 + std_b ** 2) / n)


def required_n_continuous_unpaired(std_a, std_b, target_mde, power=0.8, alpha=0.05):
    """Inverse of mde_continuous_unpaired. Returns required PER-ARM sample size."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2 * (std_a ** 2 + std_b ** 2)) / target_mde ** 2
    return int(np.ceil(n))


def compute_pilot_stats(pilot_results):
    """
    From a pilot scored under all three arms, computes the per-arm statistics
    the unpaired power analysis needs: each arm's hit rate (for the binary
    two-proportion test) and each arm's mean/std of Recall@K and NDCG@K (for
    the Welch continuous tests). These are marginal per-arm quantities -- the
    unpaired design can't exploit any playlist-level pairing -- so a single
    arm's mean/std over the pilot is exactly what a live test would work from.
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
    Per-ARM sample size required for ~80% power at each metric's target MDE, for
    each comparison (vs Control, vs ALS-only), using the independent-groups
    formulas -- what a real live A/B test needs, since it can't reuse a playlist
    across arms. Returns the per-metric-per-comparison requirements plus
    'overall' (the max across all of them -- the binding per-arm constraint,
    since every reported comparison must be powered at once).
    """
    required = {"binary": {}, "recall": {}, "ndcg": {}}
    for comparison, other_arm in (("vs_control", "control"), ("vs_als", "als")):
        required["binary"][comparison] = required_n_binary_unpaired(
            pilot_stats["hit_rate"]["treatment"], pilot_stats["hit_rate"][other_arm], target_mde_hit_rate)
        _, t_recall_std = pilot_stats["recall_arm_stats"]["treatment"]
        _, other_recall_std = pilot_stats["recall_arm_stats"][other_arm]
        required["recall"][comparison] = required_n_continuous_unpaired(
            t_recall_std, other_recall_std, target_mde_recall)
        _, t_ndcg_std = pilot_stats["ndcg_arm_stats"]["treatment"]
        _, other_ndcg_std = pilot_stats["ndcg_arm_stats"][other_arm]
        required["ndcg"][comparison] = required_n_continuous_unpaired(
            t_ndcg_std, other_ndcg_std, target_mde_ndcg)

    required["overall"] = max(
        v for metric in ("binary", "recall", "ndcg") for v in required[metric].values()
    )
    return required


def print_power_analysis(pilot_stats, required_n, per_arm_n, target_mde_hit_rate, target_mde_recall, target_mde_ndcg):
    """
    Prints, for each metric/comparison, the MDE achievable at the planned
    per-arm sample size and the per-arm sample size required to hit the target
    MDE at ~80% power. All figures are PER ARM (independent groups); the full
    simulation fetches ~3x this in total, one arm per playlist.
    """
    print("\n=== Pre-Experiment Power Analysis (between-subjects / unpaired) ===")
    print(f"Planned per-arm sample size: {per_arm_n} playlists  (~{3 * per_arm_n} total across 3 arms)")
    print(f"(Estimates from independent pilot of {PILOT_N} playlists, scored under all arms)\n")

    comparisons = (("vs_control", "vs Control", "control"), ("vs_als", "vs ALS-only", "als"))

    def _report(label, mde_fn, stat_args_fn, target_mde, req_n_by_comparison, unit_desc):
        print(f"{label}:")
        for key, comp_label, other_arm in comparisons:
            mde_at_n = mde_fn(per_arm_n, *stat_args_fn(other_arm))
            req_n = req_n_by_comparison[key]
            print(f"  {comp_label} -- MDE at per-arm n={per_arm_n}: {mde_at_n:.4f}")
            print(f"    Target effect size: {target_mde:.4f} {unit_desc} -- requires per-arm n >= {req_n}")
            if per_arm_n >= req_n:
                print(f"    [OK] per-arm n={per_arm_n} meets or exceeds the required sample size ({req_n}).")
            else:
                print(f"    [WARN] per-arm n={per_arm_n} is BELOW the required sample size ({req_n}) "
                      f"to detect a {target_mde:.4f} lift with ~80% power.")

    def _binary_args(other_arm):
        return pilot_stats["hit_rate"]["treatment"], pilot_stats["hit_rate"][other_arm]

    def _recall_args(other_arm):
        return pilot_stats["recall_arm_stats"]["treatment"][1], pilot_stats["recall_arm_stats"][other_arm][1]

    def _ndcg_args(other_arm):
        return pilot_stats["ndcg_arm_stats"]["treatment"][1], pilot_stats["ndcg_arm_stats"][other_arm][1]

    _report("Binary hit rate", mde_binary_unpaired, _binary_args,
            target_mde_hit_rate, required_n["binary"], "absolute lift")
    print()
    _report("Continuous recall", mde_continuous_unpaired, _recall_args,
            target_mde_recall, required_n["recall"], "mean difference")
    print()
    _report("NDCG@K", mde_continuous_unpaired, _ndcg_args,
            target_mde_ndcg, required_n["ndcg"], "mean NDCG difference")

    overall = required_n["overall"]
    print(f"\nOverall required per-arm n (max across metrics and comparisons): {overall}")
    if per_arm_n == overall:
        print(f"Full simulation sized from this: ~{3 * overall} playlists total (one arm per playlist).\n")
    else:
        print(f"NOTE: --n-playlists override in effect (per-arm ~{per_arm_n}, ~{3 * per_arm_n} total) "
              f"instead of the power-analysis value (per-arm {overall}, ~{3 * overall} total).\n")


# Simulation
def run_pilot(playlists, als_model, user_item_matrix, xgb_model, popularity_baseline, k=20, holdout_frac=0.2, seed=None, conn=None):
    """
    Scores every pilot playlist under ALL three arms (treatment, ALS-only,
    control) against the same held-out split, to estimate per-arm stats for the
    power analysis. The pilot is small, so running every arm on every playlist
    is cheap and maximizes the precision of each arm's mean/variance estimate.
    Unlike the full simulation, this is NOT one-arm-per-playlist.
    """
    treatment_binary, als_binary, control_binary = [], [], []
    treatment_recall, als_recall, control_recall = [], [], []
    treatment_ndcg, als_ndcg, control_ndcg = [], [], []

    for index, (pid, track_ids) in enumerate(playlists.items()):
        print(f"Pilot playlist {index + 1} of {len(playlists)}")
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


def run_simulation(playlists, als_model, user_item_matrix, xgb_model, popularity_baseline, k=20, holdout_frac=0.2, seed=None, conn=None):
    """
    Full between-subjects simulation: each playlist is assigned to EXACTLY ONE
    arm (index % 3 via ARMS -- the fetch is already seed-shuffled, so this is an
    effectively-random 3-way split) and scored only under that arm. Returns
    per-arm metric lists over DISJOINT playlist groups (~n/3 each), the way a
    real live A/B test collects data.

    Only the treatment arm runs the full engine (retrieval + rerank); ALS-only
    runs retrieval alone; control just reuses the precomputed popularity list --
    so roughly two-thirds of the expensive full-engine work is avoided vs.
    scoring every playlist under every arm.
    """
    results = {
        "treatment_binary": [], "als_binary": [], "control_binary": [],
        "treatment_recall": [], "als_recall": [], "control_recall": [],
        "treatment_ndcg": [], "als_ndcg": [], "control_ndcg": [],
    }

    for index, (pid, track_ids) in enumerate(playlists.items()):
        arm = ARMS[index % 3]
        print(f"Simulating playlist {index + 1} of {len(playlists)} [{arm}]")
        seed_tracks, held_out = split_holdout(track_ids, holdout_frac, seed=stable_seed(pid, seed))

        if arm == "treatment":
            candidate_integers, _als_scores = get_als_candidates(seed_tracks, als_model, user_item_matrix, n=200, conn=conn)
            recs = rank_candidates(seed_tracks, candidate_integers, xgb_model, k=k, conn=conn)["track_id"].to_list()
        elif arm == "als":
            recs = get_als_only_recommendations(seed_tracks, als_model, user_item_matrix, k=k, conn=conn)
        else:  # control
            recs = popularity_baseline

        results[f"{arm}_binary"].append(binary_hit(recs, held_out, k))
        results[f"{arm}_recall"].append(recall_at_k(recs, held_out, k))
        results[f"{arm}_ndcg"].append(ndcg_at_k(recs, held_out, k))

    return results


# Reporting
def _report_metric_unpaired(label, treatment, arm_b, control, arm_b_name):
    """Reports Treatment vs Control and Treatment vs `arm_b_name` for one continuous
    metric as INDEPENDENT groups (Welch's t-test)."""
    print(f"\n{label} (independent groups):")

    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
    print(f"  Treatment: {np.mean(treatment):.4f} (n={len(treatment)})   "
          f"Control: {np.mean(control):.4f} (n={len(control)})   "
          f"Lift: {np.mean(treatment) - np.mean(control):+.4f}")
    print(f"  Welch's t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
    mean_diff, lo, hi = bootstrap_ci_unpaired(treatment, control)
    print(f"  Bootstrap 95% CI vs Control: [{lo:.4f}, {hi:.4f}]  (mean diff: {mean_diff:.4f})")
    check_agreement(p_val < 0.05, lo, hi, f"{label} vs Control")

    t_stat_b, p_val_b = stats.ttest_ind(treatment, arm_b, equal_var=False)
    print(f"  {arm_b_name}: {np.mean(arm_b):.4f} (n={len(arm_b)})   "
          f"Lift over {arm_b_name}: {np.mean(treatment) - np.mean(arm_b):+.4f}")
    print(f"  Welch's t-test: t = {t_stat_b:.3f}, p = {p_val_b:.4f}")
    mean_diff_b, lo_b, hi_b = bootstrap_ci_unpaired(treatment, arm_b)
    print(f"  Bootstrap 95% CI vs {arm_b_name}: [{lo_b:.4f}, {hi_b:.4f}]  (mean diff: {mean_diff_b:.4f})")
    check_agreement(p_val_b < 0.05, lo_b, hi_b, f"{label} vs {arm_b_name}")


def _report_binary_vs_unpaired(treatment_binary, other_binary, other_name):
    """Reports Treatment vs `other_name` for the binary hit-rate metric as
    INDEPENDENT groups via `two_proportion_ztest`."""
    p_t, p_o, z_stat, p_val = two_proportion_ztest(treatment_binary, other_binary)
    print(f"  {other_name}: {p_o:.4f} (n={len(other_binary)})   Lift over {other_name}: {p_t - p_o:+.4f}")
    print(f"  Two-proportion z-test vs {other_name}: z = {z_stat:.3f}, p = {p_val:.4f}")
    mean_diff, lo, hi = bootstrap_ci_unpaired(treatment_binary, other_binary)
    print(f"  Bootstrap 95% CI vs {other_name}: [{lo:.4f}, {hi:.4f}]  (mean diff: {mean_diff:.4f})")
    check_agreement(p_val < 0.05, lo, hi, f"Hit rate vs {other_name}")


def report(results, k, required_per_arm_n=None):
    """
    Reports the between-subjects results. Each arm's lists come from a DISJOINT
    group of playlists (assigned in run_simulation), so all comparisons use
    independent-sample tests -- the way a real live A/B test is analyzed.
    """
    t_ndcg, a_ndcg, c_ndcg = results["treatment_ndcg"], results["als_ndcg"], results["control_ndcg"]
    t_recall, a_recall, c_recall = results["treatment_recall"], results["als_recall"], results["control_recall"]
    t_bin, a_bin, c_bin = results["treatment_binary"], results["als_binary"], results["control_binary"]

    print("\n=== A/B/C Test Simulation Results: Full Engine vs ALS-only vs Popularity ===\n")
    print("(Between-subjects: each playlist was assigned to exactly one arm -- disjoint groups,")
    print(" the way a real live A/B test randomizes users -- so independent-sample tests are used")
    print(" rather than the paired tests in evaluate.py.)")
    print(f"\nGroup sizes -- Treatment: {len(t_ndcg)}   Control: {len(c_ndcg)}   ALS-only: {len(a_ndcg)}")

    print(f"\n[PRIMARY] NDCG@{k} (matches XGBRanker training objective):")
    _report_metric_unpaired("NDCG", t_ndcg, a_ndcg, c_ndcg, "ALS-only")

    print(f"\n[SECONDARY] Recall@{k} (fraction of held-out tracks recovered):")
    _report_metric_unpaired("Recall", t_recall, a_recall, c_recall, "ALS-only")

    print(f"\n[TERTIARY] Binary hit rate (>=1 relevant track in Top {k}):")
    print(f"  Treatment: {np.mean(t_bin):.4f} (n={len(t_bin)})")
    _report_binary_vs_unpaired(t_bin, c_bin, "Control")
    _report_binary_vs_unpaired(t_bin, a_bin, "ALS-only")

    if required_per_arm_n is not None:
        smallest = min(len(t_ndcg), len(a_ndcg), len(c_ndcg))
        print(f"\nSmallest arm group: {smallest} playlists; the power analysis required per-arm "
              f"n >= {required_per_arm_n} for the target MDE.")
        if smallest < required_per_arm_n:
            print("[WARN] At least one arm is underpowered relative to that requirement.")


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-playlists", type=int, default=None,
                         help="Override the TOTAL sample size across all arms (each arm gets ~1/3), "
                              "instead of deriving it from the pilot's power analysis. If it implies "
                              "fewer than the required per-arm n, the run proceeds but is flagged as "
                              "underpowered.")
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
    # a real experiment wouldn't reuse pilot traffic in the final analysis). Scored
    # under all three arms, since the power analysis needs each arm's marginal
    # mean/variance and the pilot is small enough to afford it.
    print(f"Running pilot on {args.pilot_n} playlists (all arms) to estimate per-arm statistics...")
    pilot_playlists = fetch_playlists(conn, args.pilot_n, seed=args.seed)
    pilot_results = run_pilot(
        pilot_playlists, als_model, user_item_matrix, xgb_model,
        popularity_baseline, k=args.k, holdout_frac=args.holdout_frac, seed=args.seed, conn=conn,
    )
    pilot_stats = compute_pilot_stats(pilot_results)

    # Step 2: power analysis -- gives the required PER-ARM n; the full sample
    # (one arm per playlist) needs ~3x that in total.
    required_n = compute_required_n(
        pilot_stats, args.target_mde_hit_rate, args.target_mde_recall, args.target_mde_ndcg,
    )
    if args.n_playlists is not None:
        total_n = args.n_playlists
        per_arm_n = total_n // 3
    else:
        per_arm_n = required_n["overall"]
        total_n = 3 * per_arm_n
    print_power_analysis(
        pilot_stats, required_n, per_arm_n,
        args.target_mde_hit_rate, args.target_mde_recall, args.target_mde_ndcg,
    )

    # Step 3: full simulation, one arm per playlist, sized by the power analysis above
    print("=== Running Full Simulation (one arm per playlist) ===\n")
    all_playlists = fetch_playlists(conn, total_n, seed=args.seed + 1)
    results = run_simulation(
        all_playlists, als_model, user_item_matrix, xgb_model,
        popularity_baseline, k=args.k, holdout_frac=args.holdout_frac, seed=args.seed + 1, conn=conn,
    )

    # Step 4: results
    report(results, args.k, required_per_arm_n=required_n["overall"])

    conn.close()
