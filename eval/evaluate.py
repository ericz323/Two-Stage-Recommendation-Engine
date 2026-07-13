"""
evaluate.py
Offline evaluation of the Two-Stage Playlist Engine using a held-out
split of tracks per playlist.

Compares three systems:
  1. Full engine  -- ALS retrieval + XGBRanker reranking (the real pipeline)
  2. ALS-only     -- ALS retrieval, no reranking (isolates the ranker's contribution)
  3. Popularity   -- most-interacted-with tracks globally (sanity-check floor)

The Full Engine vs ALS-only comparison is the most informative one: it tells
you whether the reranking stage is actually improving results over raw
retrieval, or just adding complexity.

Also reports a POPULARITY-DEBIASED view: the same metrics restricted to the
long-tail held-out tracks (those outside the global popular head). Aggregate
recall/NDCG is inflated by popularity bias -- popular tracks dominate held-out
sets and the popularity baseline recovers them "for free" -- so the long-tail
view isolates whether personalization adds value where popularity cannot.

Run:
    python eval/evaluate.py --n-playlists 2000 --holdout-frac 0.2 --k 20 --n-popular 10000
"""

import argparse
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg
import xgboost as xgb
from implicit.als import AlternatingLeastSquares
from scipy import stats
from scipy.sparse import load_npz

from src.recommend import get_als_candidates, rank_candidates, get_als_only_recommendations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DB_URI = 'postgresql://postgres:postgres123@localhost:5432/playlist_engine'


def fetch_playlists(conn, n_playlists, min_tracks=10, seed=None):
    """
    Pull a sample of playlists with enough tracks to do a meaningful split.
    If `seed` is given, Postgres's RNG is seeded first so ORDER BY random()
    draws the same sample (in the same order) on every run; otherwise each
    call draws a fresh, unreproducible sample.
    """
    query = """
        SELECT playlist_id, ARRAY_AGG(track_id) AS track_ids
        FROM interaction_matrix
        GROUP BY playlist_id
        HAVING COUNT(*) >= %s
        ORDER BY random()
        LIMIT %s;
    """
    with conn.cursor() as cur:
        if seed is not None:
            cur.execute("SELECT setseed(%s)", (_pg_seed(seed),))
        cur.execute(query, (min_tracks, n_playlists))
        rows = cur.fetchall()
    return {pid: tracks for pid, tracks in rows}


def _pg_seed(seed):
    """Maps an arbitrary int seed to the [-1, 1] float Postgres's setseed() requires."""
    return (seed % 10_000) / 5_000.0 - 1


def stable_seed(pid, base_seed=0):
    """
    Deterministic per-playlist seed for split_holdout. Python's built-in hash()
    is randomized per-process for str/tuple keys (PYTHONHASHSEED), so using it
    directly means the same playlist gets a different holdout split on every
    run. crc32 has no such randomization, so (pid, base_seed) always maps to
    the same value.
    """
    return zlib.crc32(f"{base_seed}:{pid}".encode())


def fetch_global_popularity(conn, top_n=20):
    """Baseline: most popular tracks overall, by interaction count."""
    query = """
        SELECT track_id, COUNT(*) AS cnt
        FROM interaction_matrix
        GROUP BY track_id
        ORDER BY cnt DESC
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (top_n,))
        rows = cur.fetchall()
    return [r[0] for r in rows]


def fetch_popular_track_set(conn, n_popular):
    """
    Returns the SET of the top-`n_popular` track_ids by interaction count -- the
    "head" of the catalog. Used to debias evaluation: a held-out track in this set
    is "popular", the rest are "long-tail". On long-tail held-out tracks the
    popularity baseline structurally can't score, so any lift there reflects real
    personalization rather than the popularity-bias of the held-out split.
    """
    query = """
        SELECT track_id
        FROM interaction_matrix
        GROUP BY track_id
        ORDER BY COUNT(*) DESC
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (n_popular,))
        return {r[0] for r in cur.fetchall()}


def split_holdout(track_ids, holdout_frac=0.2, seed=None):
    """Randomly hide a fraction of a playlist's tracks."""
    rng = np.random.default_rng(seed)
    tracks = list(track_ids)
    rng.shuffle(tracks)
    n_hold = max(1, int(len(tracks) * holdout_frac))
    held_out = tracks[:n_hold]
    seed_tracks = tracks[n_hold:]
    return seed_tracks, held_out


def recall_at_k(recommended, held_out, k=20):
    rec_set = set(recommended[:k])
    hits = len(rec_set & set(held_out))
    return hits / len(held_out) if held_out else 0.0


def ndcg_at_k(recommended, held_out, k=20):
    held_out_set = set(held_out)
    dcg = 0.0
    for i, track in enumerate(recommended[:k]):
        if track in held_out_set:
            dcg += 1.0 / np.log2(i + 2)  # rank is 1-indexed, +2 to avoid log(1)=0
    ideal_hits = min(len(held_out_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def get_engine_and_als_recommendations(seed_track_ids, als_model, user_item_matrix, xgb_model, k=20, conn=None):
    """
    Runs Stage 1 (ALS retrieval) ONCE and derives both the full engine's
    reranked recommendations and the ALS-only ablation's recommendations from
    it, instead of retrieving twice for the same seed tracks -- ALS scores
    every item once and returns the top-N by score, so the ALS-only arm's
    top-k is always identical to the head of the full engine's larger
    candidate pool.

    Returns: (recs_full, recs_als), both ranked lists of track_id strings, length k.
    """
    candidate_integers, als_scores = get_als_candidates(seed_track_ids, als_model, user_item_matrix, n=200, conn=conn)
    df_recs = rank_candidates(seed_track_ids, candidate_integers, als_scores, xgb_model, k=k, conn=conn)
    recs_full = df_recs['track_id'].to_list()
    recs_als = get_als_only_recommendations(
        seed_track_ids, als_model, user_item_matrix, k=k, candidate_integers=candidate_integers, conn=conn)
    return recs_full, recs_als


def evaluate_pipeline(playlists, als_model, user_item_matrix, xgb_model, k=20,
                       holdout_frac=0.2, popularity_baseline=None, popular_set=None,
                       seed=None, conn=None):
    """
    Runs the full two-stage engine, the ALS-only baseline (no reranking),
    and a popularity baseline over the same held-out splits.

    If `popular_set` is given, ALSO computes popularity-debiased metrics on the
    LONG-TAIL held-out tracks only (those not in the popular set): the same
    recall/NDCG restricted to that subset, `*_longtail` keys. Only playlists with
    at least one long-tail held-out track contribute, so all arms stay paired.
    """
    results = defaultdict(list)
    n_popular_heldout = 0
    n_longtail_heldout = 0

    for index, (pid, track_ids) in enumerate(playlists.items()):
        print(f"Evaluating playlist {index + 1} of {len(playlists)}")
        seed_tracks, held_out = split_holdout(track_ids, holdout_frac, seed=stable_seed(pid, seed))

        # --- Full two-stage engine + ALS-only (retrieval, no reranking) ---
        recs_full, recs_als = get_engine_and_als_recommendations(
            seed_tracks, als_model, user_item_matrix, xgb_model, k=k, conn=conn)
        results["engine_recall"].append(recall_at_k(recs_full, held_out, k))
        results["engine_ndcg"].append(ndcg_at_k(recs_full, held_out, k))
        results["als_only_recall"].append(recall_at_k(recs_als, held_out, k))
        results["als_only_ndcg"].append(ndcg_at_k(recs_als, held_out, k))

        # --- Popularity baseline ---
        if popularity_baseline is not None:
            results["baseline_recall"].append(recall_at_k(popularity_baseline, held_out, k))
            results["baseline_ndcg"].append(ndcg_at_k(popularity_baseline, held_out, k))

        # --- Popularity-debiased: metrics on the long-tail held-out subset only ---
        if popular_set is not None:
            held_out_longtail = [t for t in held_out if t not in popular_set]
            n_popular_heldout += len(held_out) - len(held_out_longtail)
            n_longtail_heldout += len(held_out_longtail)
            if held_out_longtail:
                results["engine_recall_longtail"].append(recall_at_k(recs_full, held_out_longtail, k))
                results["engine_ndcg_longtail"].append(ndcg_at_k(recs_full, held_out_longtail, k))
                results["als_only_recall_longtail"].append(recall_at_k(recs_als, held_out_longtail, k))
                results["als_only_ndcg_longtail"].append(ndcg_at_k(recs_als, held_out_longtail, k))
                if popularity_baseline is not None:
                    results["baseline_recall_longtail"].append(recall_at_k(popularity_baseline, held_out_longtail, k))
                    results["baseline_ndcg_longtail"].append(ndcg_at_k(popularity_baseline, held_out_longtail, k))

    # Stash the held-out composition (single-element lists to fit the defaultdict(list)).
    results["_n_popular_heldout"] = [n_popular_heldout]
    results["_n_longtail_heldout"] = [n_longtail_heldout]
    return results


def summarize(results):
    print("\n=== Offline Evaluation Results ===\n")
    print(f"{'Metric':<20}{'Full Engine':<14}{'ALS-only':<14}{'Popularity':<12}")
    for metric in ["recall", "ndcg"]:
        engine_vals = results[f"engine_{metric}"]
        als_vals = results.get(f"als_only_{metric}", [])
        base_vals = results.get(f"baseline_{metric}", [])
        engine_mean = np.mean(engine_vals)
        als_mean = np.mean(als_vals) if als_vals else float("nan")
        base_mean = np.mean(base_vals) if base_vals else float("nan")
        print(f"{metric.upper():<20}{engine_mean:<14.4f}{als_mean:<14.4f}{base_mean:<12.4f}")

    # --- Most important comparison: does reranking actually help? ---
    if results.get("als_only_ndcg"):
        print("\n[KEY TEST] Full Engine vs ALS-only (does XGBRanker add value over raw retrieval?)")
        t_stat, p_val = stats.ttest_rel(results["engine_ndcg"], results["als_only_ndcg"])
        lift = np.mean(results["engine_ndcg"]) - np.mean(results["als_only_ndcg"])
        print(f"  NDCG lift from reranking: {lift:+.4f}  (t={t_stat:.3f}, p={p_val:.4f})")
        if p_val >= 0.05:
            print("  NOTE: not statistically significant -- reranking may not be earning its complexity here.")

    # --- Secondary: does the full engine beat a trivial popularity baseline? ---
    if results.get("baseline_recall"):
        t_stat, p_val = stats.ttest_rel(results["engine_recall"], results["baseline_recall"])
        print(f"\nFull Engine vs Popularity -- Paired t-test on Recall@K: t={t_stat:.3f}, p={p_val:.4f}")

    if results.get("baseline_ndcg"):
        t_stat, p_val = stats.ttest_rel(results["engine_ndcg"], results["baseline_ndcg"])
        print(f"Full Engine vs Popularity -- Paired t-test on NDCG@K:  t={t_stat:.3f}, p={p_val:.4f}")

    _summarize_debiased(results)


def _summarize_debiased(results):
    """
    Reports popularity-debiased metrics: recall/NDCG on the LONG-TAIL held-out
    tracks only. The popularity baseline structurally cannot recover long-tail
    tracks (it only recommends the head), so this isolates whether personalization
    adds value where popularity can't -- the comparison the aggregate metric hides.
    """
    if not results.get("engine_recall_longtail"):
        return

    n_lt = len(results["engine_recall_longtail"])
    pop_ct = results.get("_n_popular_heldout", [0])[0]
    lt_ct = results.get("_n_longtail_heldout", [0])[0]
    total_ct = pop_ct + lt_ct

    print("\n=== Popularity-Debiased Results (long-tail held-out tracks only) ===")
    print("Restricted to held-out tracks NOT in the global popular set -- where the popularity")
    print("baseline structurally can't score, so any lift reflects real personalization.")
    if total_ct:
        print(f"Held-out composition: {pop_ct} popular / {lt_ct} long-tail "
              f"({100 * lt_ct / total_ct:.1f}% long-tail); {n_lt} playlists have >=1 long-tail held-out track.")

    print(f"\n{'Metric':<26}{'Full Engine':<14}{'ALS-only':<14}{'Popularity':<12}")
    for metric in ["recall", "ndcg"]:
        engine_vals = results[f"engine_{metric}_longtail"]
        als_vals = results.get(f"als_only_{metric}_longtail", [])
        base_vals = results.get(f"baseline_{metric}_longtail", [])
        engine_mean = np.mean(engine_vals)
        als_mean = np.mean(als_vals) if als_vals else float("nan")
        base_mean = np.mean(base_vals) if base_vals else float("nan")
        print(f"{(metric.upper() + ' (long-tail)'):<26}{engine_mean:<14.4f}{als_mean:<14.4f}{base_mean:<12.4f}")

    if results.get("als_only_ndcg_longtail"):
        print("\n[KEY TEST -- debiased] Full Engine vs ALS-only on long-tail NDCG")
        t_stat, p_val = stats.ttest_rel(results["engine_ndcg_longtail"], results["als_only_ndcg_longtail"])
        lift = np.mean(results["engine_ndcg_longtail"]) - np.mean(results["als_only_ndcg_longtail"])
        print(f"  NDCG lift from reranking (long-tail): {lift:+.4f}  (t={t_stat:.3f}, p={p_val:.4f})")

    if results.get("baseline_recall_longtail"):
        t_stat, p_val = stats.ttest_rel(results["engine_recall_longtail"], results["baseline_recall_longtail"])
        print(f"\nFull Engine vs Popularity (long-tail) -- Recall@K: t={t_stat:.3f}, p={p_val:.4f}")
        t_stat, p_val = stats.ttest_rel(results["engine_ndcg_longtail"], results["baseline_ndcg_longtail"])
        print(f"Full Engine vs Popularity (long-tail) -- NDCG@K:  t={t_stat:.3f}, p={p_val:.4f}")


def load_models():
    print('Loading models into memory...')
    als_model = AlternatingLeastSquares().load(str(MODELS_DIR / 'als_model.npz'))
    user_item_matrix = load_npz(str(MODELS_DIR / 'user_item_matrix.npz'))
    xgb_model = xgb.XGBRanker()
    xgb_model.load_model(str(MODELS_DIR / 'xgb_ranker.json'))
    return als_model, user_item_matrix, xgb_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-playlists", type=int, default=2000)
    parser.add_argument("--holdout-frac", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed for playlist sampling and holdout splits, for reproducible "
                              "runs. Pass a different value (or write your own randomization) to "
                              "draw a fresh sample.")
    parser.add_argument("--n-popular", type=int, default=50,
                         help="Size of the global 'popular head' for the popularity-debiased eval. "
                              "Held-out tracks outside the top-N most-interacted tracks are treated "
                              "as long-tail, where the popularity baseline can't score (default: 10000).")
    args = parser.parse_args()

    als_model, user_item_matrix, xgb_model = load_models()

    # prepare_threshold=None: this connection is shared across the whole run, so the
    # same query text runs thousands of times on it. psycopg3 would auto-prepare
    # each after 5 executions, after which PostgreSQL can switch to a GENERIC plan
    # for `col = ANY($1)` that can't see the array contents, misestimates
    # selectivity, and seq-scans despite the indexes. Disabling preparation keeps
    # every execute planned with the actual values (custom plan) -> uses the index.
    # autocommit is fine for a read-only connection but is NOT the perf fix.
    conn = psycopg.connect(DB_URI, autocommit=True, prepare_threshold=None)
    print("Fetching playlists...")
    playlists = fetch_playlists(conn, args.n_playlists, seed=args.seed)
    print("Creating popularity baseline...")
    popularity_baseline = fetch_global_popularity(conn, top_n=args.k)
    print(f"Fetching popular head ({args.n_popular} tracks) for debiased eval...")
    popular_set = fetch_popular_track_set(conn, args.n_popular)

    print("Evaluating recommendations...")
    results = evaluate_pipeline(
        playlists,
        als_model,
        user_item_matrix,
        xgb_model,
        k=args.k,
        holdout_frac=args.holdout_frac,
        popularity_baseline=popularity_baseline,
        popular_set=popular_set,
        seed=args.seed,
        conn=conn,
    )
    summarize(results)
    conn.close()


if __name__ == "__main__":
    main()
