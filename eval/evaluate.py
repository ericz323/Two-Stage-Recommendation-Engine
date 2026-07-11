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

Run:
    python eval/evaluate.py --n-playlists 2000 --holdout-frac 0.2 --k 20
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

from src.recommend import generate_recommendations, get_als_only_recommendations

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


def get_recommendations(seed_track_ids, als_model, user_item_matrix, xgb_model, k=20):
    """
    Full engine: ALS retrieval (Stage 1) + XGBRanker reranking (Stage 2).
    Returns a ranked list of track_id strings, length k.
    """
    df_recs = generate_recommendations(seed_track_ids, als_model, user_item_matrix, xgb_model, k=k)
    return df_recs['track_id'].to_list()


def evaluate_pipeline(playlists, als_model, user_item_matrix, xgb_model, k=20,
                       holdout_frac=0.2, popularity_baseline=None, seed=None):
    """
    Runs the full two-stage engine, the ALS-only baseline (no reranking),
    and a popularity baseline over the same held-out splits.
    """
    results = defaultdict(list)

    for index, (pid, track_ids) in enumerate(playlists.items()):
        print(f"Evaluating playlist {index + 1} of {len(playlists)}")
        seed_tracks, held_out = split_holdout(track_ids, holdout_frac, seed=stable_seed(pid, seed))

        # --- Full two-stage engine (ALS retrieval + XGBRanker reranking) ---
        recs_full = get_recommendations(seed_tracks, als_model, user_item_matrix, xgb_model, k=k)
        results["engine_recall"].append(recall_at_k(recs_full, held_out, k))
        results["engine_ndcg"].append(ndcg_at_k(recs_full, held_out, k))

        # --- ALS-only (retrieval, no reranking) ---
        recs_als = get_als_only_recommendations(seed_tracks, als_model, user_item_matrix, k=k)
        results["als_only_recall"].append(recall_at_k(recs_als, held_out, k))
        results["als_only_ndcg"].append(ndcg_at_k(recs_als, held_out, k))

        # --- Popularity baseline ---
        if popularity_baseline is not None:
            results["baseline_recall"].append(recall_at_k(popularity_baseline, held_out, k))
            results["baseline_ndcg"].append(ndcg_at_k(popularity_baseline, held_out, k))

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
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed for playlist sampling and holdout splits, for reproducible "
                              "runs. Pass a different value (or write your own randomization) to "
                              "draw a fresh sample.")
    args = parser.parse_args()

    als_model, user_item_matrix, xgb_model = load_models()

    conn = psycopg.connect(DB_URI)
    print("Fetching playlists...")
    playlists = fetch_playlists(conn, args.n_playlists, seed=args.seed)
    print("Creating popularity baseline...")
    popularity_baseline = fetch_global_popularity(conn, top_n=args.k)

    print("Evaluating recommendations...")
    results = evaluate_pipeline(
        playlists,
        als_model,
        user_item_matrix,
        xgb_model,
        k=args.k,
        holdout_frac=args.holdout_frac,
        popularity_baseline=popularity_baseline,
        seed=args.seed,
    )
    summarize(results)
    conn.close()


if __name__ == "__main__":
    main()
