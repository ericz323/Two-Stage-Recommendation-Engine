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
    get_recommendations,
    load_models,
)
from src.recommend import get_als_only_recommendations

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
    }


def compute_required_n(pilot_stats, target_mde_hit_rate, target_mde_recall, target_mde_ndcg):
    """
    Computes the sample size required to hit ~80% power for each metric's
    target MDE, for each comparison (vs Control, vs ALS-only), using the
    paired-design pilot stats. Returns the per-metric-per-comparison
    requirements plus the overall requirement (the max across all of them --
    the binding constraint, since the full simulation must be large enough
    to power every reported comparison at once).
    """
    required = {"binary": {}, "recall": {}, "ndcg": {}}
    for comparison in ("vs_control", "vs_als"):
        required["binary"][comparison] = required_n_binary_paired(
            pilot_stats["binary_discordant_rate"][comparison], target_mde_hit_rate)
        required["recall"][comparison] = required_n_continuous_paired(
            pilot_stats["recall_diff_std"][comparison], target_mde_recall)
        required["ndcg"][comparison] = required_n_continuous_paired(
            pilot_stats["ndcg_diff_std"][comparison], target_mde_ndcg)

    required["overall"] = max(
        v for metric in ("binary", "recall", "ndcg") for v in required[metric].values()
    )
    return required


def print_power_analysis(pilot_stats, required_n, n_playlists, target_mde_hit_rate, target_mde_recall, target_mde_ndcg):
    """
    Prints MDE for each metric/comparison given the full simulation sample
    size, plus the sample size required to detect the target effect size --
    so the "how many playlists do we need" decision is explicit rather than
    reverse-engineered from whatever n_playlists happened to be passed in.
    """
    print("\n=== Pre-Experiment Power Analysis ===")
    print(f"Full simulation sample size: {n_playlists} playlists per arm")
    print(f"(Estimates from independent pilot of {PILOT_N} playlists, paired-design formulas)\n")

    comparisons = (("vs_control", "vs Control"), ("vs_als", "vs ALS-only"))

    def _report(label, mde_fn, stat_key, target_mde, req_n_by_comparison, unit_desc):
        print(f"{label}:")
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

    _report("Binary hit rate", mde_binary_paired, "binary_discordant_rate",
             target_mde_hit_rate, required_n["binary"], "absolute lift")
    print()
    _report("Continuous recall", mde_continuous_paired, "recall_diff_std",
             target_mde_recall, required_n["recall"], "mean difference")
    print()
    _report("NDCG@K", mde_continuous_paired, "ndcg_diff_std",
             target_mde_ndcg, required_n["ndcg"], "mean NDCG difference")

    print(f"\nOverall required n (max across metrics and comparisons): {required_n['overall']} playlists/arm")
    if n_playlists == required_n["overall"]:
        print("Full simulation sample size was set from this power analysis.\n")
    else:
        print(f"NOTE: --n-playlists override was used ({n_playlists}) instead of the "
              f"power-analysis-derived value ({required_n['overall']}).\n")


# Simulation
def run_simulation(playlists, als_model, user_item_matrix, xgb_model, popularity_baseline, k=20, holdout_frac=0.2, seed=None):
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

        recs_treatment = get_recommendations(seed_tracks, als_model, user_item_matrix, xgb_model, k=k)
        recs_als = get_als_only_recommendations(seed_tracks, als_model, user_item_matrix, k=k)
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
        popularity_baseline, k=args.k, holdout_frac=args.holdout_frac, seed=args.seed,
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
        popularity_baseline, k=args.k, holdout_frac=args.holdout_frac, seed=args.seed + 1,
    )

    # Step 4: results
    report(results, len(all_playlists), args.k)

    conn.close()
