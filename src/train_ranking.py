"""
train_ranking.py
Trains the Stage 2 XGBRanker that reranks the ALS retrieval candidates.

Trained the SAME way it is served ("train like you serve"). For each playlist we
hide a fraction of its tracks (the TARGET), retrieve candidates from the
remaining SEED tracks via the exact cold-start path inference uses
(recalculate_user), and label each candidate 1 if it is a hidden target and 0
otherwise. The taste profile (avg_*) is computed over the SEED tracks only (never
the labeled targets), and als_score is the recalculate_user retrieval score --
identical in origin to the als_score feature at inference. So there is no profile
circularity and no train/serve skew in als_score.

Two negative-sampling strategies can be trained and saved separately, to compare
side-by-side. Everything else is held identical -- same playlists, same
seed/target split, same positives, same features -- so this isolates the
negative-sampling variable:
  - hard   : negatives are ALS-retrieved tracks that are NOT held-out targets
             (the pool the ranker actually reorders at serving).
  - random : negatives are random catalog tracks not in the playlist.

Because positives are only the retrieved held-out targets, most playlists yield
few (or zero) positives; playlists with none are dropped. Increase --n-playlists
or --n-candidates for more positive coverage.

Run:
    python src/train_ranking.py --negatives both
    python src/train_ranking.py --negatives hard --out models/xgb_ranker.json
"""

import argparse
import zlib
from collections import defaultdict

import numpy as np
import polars as pl
import psycopg
import xgboost as xgb
from implicit.als import AlternatingLeastSquares
from pathlib import Path
from scipy.sparse import csr_matrix, load_npz

DB_URI = "postgresql://postgres:postgres123@localhost:5432/playlist_engine"
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ -> project root
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

N_TRAIN_PLAYLISTS = 10000  # playlists sampled for training
N_CANDIDATES = 200         # ALS candidates retrieved per playlist (matches inference pool)
HOLDOUT_FRAC = 0.2         # fraction of each playlist hidden as targets (matches eval)
N_NEG_PER_PLAYLIST = 50    # negatives kept per playlist (per strategy)
SEED = 42                  # seed for the per-playlist seed/target split + random negatives

TRACK_FEATURES = [
    'track_danceability',
    'track_tempo',
    'track_energy',
    'track_acousticness',
    'track_loudness',
    'track_valence',
]
AVG_FEATURES = [
    'avg_danceability',
    'avg_tempo',
    'avg_energy',
    'avg_acousticness',
    'avg_loudness',
    'avg_valence',
]
DIFF_FEATURES = [
    'danceability_diff',
    'tempo_diff',
    'energy_diff',
    'acousticness_diff',
    'loudness_diff',
    'valence_diff',
]
# Order MUST match feature_columns in src/recommend.py (rank_candidates).
FEATURE_COLUMNS = TRACK_FEATURES + DIFF_FEATURES + ['als_score']

# Cache for whether implicit's recommend() supports the items= kwarg (used to score
# random negatives under the recalculated user factor). None = not yet probed.
_ITEMS_PARAM_OK = None


# Local copies of eval/evaluate.py's split_holdout + stable_seed -- kept local so
# `python src/train_ranking.py` works (importing eval from src would break the path).
def _stable_seed(pid, base_seed=0):
    return zlib.crc32(f"{base_seed}:{pid}".encode())


def _split_holdout(items, holdout_frac, seed):
    rng = np.random.default_rng(seed)
    items = list(items)
    rng.shuffle(items)
    n_hold = max(1, int(len(items) * holdout_frac))
    held_out = items[:n_hold]
    seed_items = items[n_hold:]
    return seed_items, held_out


def load_als():
    """Loads the trained ALS model + user-item matrix needed to retrieve candidates."""
    als_path = MODELS_DIR / 'als_model.npz'
    matrix_path = MODELS_DIR / 'user_item_matrix.npz'
    if not als_path.exists() or not matrix_path.exists():
        raise FileNotFoundError(
            f"ALS artifacts not found in {MODELS_DIR}. Run src/train_als.py before training the ranker "
            "(candidate generation needs the retrieval model)."
        )
    als_model = AlternatingLeastSquares().load(str(als_path))
    user_item_matrix = load_npz(str(matrix_path))
    return als_model, user_item_matrix


def _sample_random_negatives(rng, n_items, exclude, n_neg):
    """Rejection-sample up to n_neg distinct track_ints in [0, n_items) not in `exclude`."""
    out, seen, attempts = [], set(), 0
    while len(out) < n_neg and attempts < n_neg * 20:
        t = int(rng.integers(0, n_items))
        attempts += 1
        if t in exclude or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return np.asarray(out, dtype=np.int64)


def _score_items_recalc(als_model, seed_row, item_tints, pid):
    """
    ALS score for `item_tints` under the SAME recalculated user factor inference
    uses -- via recommend(items=...). Falls back to the trained playlist factor's
    dot product if this implicit build has no items= kwarg (a minor train/serve
    skew for the random negatives only).
    """
    global _ITEMS_PARAM_OK
    item_tints = np.asarray(item_tints)
    if len(item_tints) == 0:
        return np.asarray([], dtype=float)

    if _ITEMS_PARAM_OK is not False:
        try:
            ids, scores = als_model.recommend(
                0, seed_row, items=item_tints, recalculate_user=True,
                N=len(item_tints), filter_already_liked_items=False)
            score_map = {int(i): float(s)
                         for i, s in zip(np.asarray(ids).reshape(-1), np.asarray(scores).reshape(-1))}
            if _ITEMS_PARAM_OK is None:
                _ITEMS_PARAM_OK = True
                print("[info] random-neg als_score via recalculate_user items= scoring.")
            return np.asarray([score_map.get(int(t), 0.0) for t in item_tints], dtype=float)
        except TypeError:
            _ITEMS_PARAM_OK = False
            print("[info] implicit recommend() has no items= kwarg; random-neg als_score "
                  "falls back to trained-factor dot product.")

    return (als_model.user_factors[pid] * als_model.item_factors[item_tints]).sum(axis=1)


def _append_candidates(acc, pid, pos_tints, pos_scores, neg_tints, neg_scores):
    """Append one playlist's positives (label 1) and negatives (label 0) to an accumulator."""
    for t, s in zip(pos_tints, pos_scores):
        acc['pl'].append(pid); acc['tint'].append(int(t)); acc['label'].append(1); acc['als'].append(float(s))
    for t, s in zip(neg_tints, neg_scores):
        acc['pl'].append(pid); acc['tint'].append(int(t)); acc['label'].append(0); acc['als'].append(float(s))


def _build_strategy_df(acc, df_map, df_metadata, df_profiles):
    """Assemble one strategy's candidate rows into the feature frame (map ids, join metadata + profile)."""
    return (
        pl.DataFrame({
            'playlist_int_id': acc['pl'],
            'track_int_id': acc['tint'],
            'label': acc['label'],
            'als_score': acc['als'],
        })
        .with_columns(pl.col('track_int_id').cast(pl.Int64), pl.col('als_score').cast(pl.Float64))
        .join(df_map, on='track_int_id', how='inner')
        .drop('track_int_id')
        .join(df_metadata, on='track_id', how='inner')       # candidate track_* features
        .join(df_profiles, on='playlist_int_id', how='inner')  # avg_* seed profile
    )


def fetch_training_data(als_model, user_item_matrix, conn, strategies,
                        n_playlists=N_TRAIN_PLAYLISTS, n_candidates=N_CANDIDATES,
                        holdout_frac=HOLDOUT_FRAC, n_neg=N_NEG_PER_PLAYLIST, seed=SEED):
    """
    Builds one labeled training frame per requested strategy. The seed/target
    split, retrieval, positives, and seed profile are shared across strategies;
    only the negatives differ (hard = retrieved non-targets, random = random
    catalog non-members). So every strategy trains on the same playlists and the
    same positives -- a clean single-variable comparison.

    return: dict {strategy: polars df} with columns playlist_int_id, label,
            als_score, the six track_*, and the six avg_* features.
    """
    print('Fetching track metadata catalog...')
    query_tracks = """
        SELECT
            track_id,
            danceability AS track_danceability,
            tempo AS track_tempo,
            energy AS track_energy,
            acousticness AS track_acousticness,
            loudness AS track_loudness,
            valence AS track_valence
        FROM track_metadata
    """
    df_metadata = pl.read_database_uri(query=query_tracks, uri=DB_URI, engine='connectorx')

    with conn.cursor() as cur:
        print(f'Sampling {n_playlists} training playlists...')
        cur.execute(
            """SELECT DISTINCT playlist_int_id
               FROM interaction_matrix_mapped
               ORDER BY playlist_int_id
               LIMIT %s""",
            (n_playlists,),
        )
        playlist_ids = [r[0] for r in cur.fetchall()]

        print('Fetching playlist tracks...')
        cur.execute(
            """SELECT playlist_int_id, track_int_id, track_id
               FROM interaction_matrix_mapped
               WHERE playlist_int_id = ANY(%s)""",
            (playlist_ids,),
        )
        track_rows = cur.fetchall()

    playlist_tracks = defaultdict(list)
    for pid, tint, tid in track_rows:
        playlist_tracks[pid].append((tint, tid))

    want_hard = 'hard' in strategies
    want_random = 'random' in strategies
    print(f'Retrieving candidates from seed tracks (recalculate_user) for {len(playlist_tracks)} playlists '
          f'[strategies: {", ".join(sorted(strategies))}]...')
    n_items = user_item_matrix.shape[1]
    rng = np.random.default_rng(seed)
    seed_pl, seed_tid = [], []                                   # (playlist, seed track_id) -> seed profile
    acc = {s: {'pl': [], 'tint': [], 'label': [], 'als': []} for s in strategies}
    kept = 0
    for pid, tracks in playlist_tracks.items():
        if len(tracks) < 2:
            continue  # need >=1 seed and >=1 target
        seed_pairs, target_pairs = _split_holdout(tracks, holdout_frac, seed=_stable_seed(pid, seed))
        if not seed_pairs:
            continue
        seed_tints = [t[0] for t in seed_pairs]
        target_tints = {int(t[0]) for t in target_pairs}
        member_tints = {int(t[0]) for t in tracks}  # seed + target -- excluded from random negatives

        # Seed user vector + retrieval -- the identical Stage-1 path used at serving.
        seed_row = csr_matrix(
            (np.ones(len(seed_tints)), (np.zeros(len(seed_tints)), np.asarray(seed_tints))),
            shape=(1, n_items),
        )
        ids, scores = als_model.recommend(
            0, seed_row, N=n_candidates, recalculate_user=True, filter_already_liked_items=True)
        ids = np.asarray(ids).reshape(-1)
        scores = np.asarray(scores).reshape(-1)
        valid = ids >= 0
        ids, scores = ids[valid], scores[valid]
        if len(ids) == 0:
            continue

        labels = np.fromiter((1 if int(t) in target_tints else 0 for t in ids), dtype=int, count=len(ids))
        pos_idx = np.where(labels == 1)[0]
        if len(pos_idx) == 0:
            continue  # no held-out target retrieved -> no positive -> skip (for ALL strategies)

        pos_tints, pos_scores = ids[pos_idx], scores[pos_idx]
        for _tint, tid in seed_pairs:
            seed_pl.append(pid)
            seed_tid.append(tid)

        if want_hard:
            neg_idx = np.where(labels == 0)[0][:n_neg]  # hardest (highest-scored) retrieved non-targets
            _append_candidates(acc['hard'], pid, pos_tints, pos_scores, ids[neg_idx], scores[neg_idx])
        if want_random:
            rand_tints = _sample_random_negatives(rng, n_items, member_tints, n_neg)
            rand_scores = _score_items_recalc(als_model, seed_row, rand_tints, pid)
            _append_candidates(acc['random'], pid, pos_tints, pos_scores, rand_tints, rand_scores)
        kept += 1

    if kept == 0:
        raise RuntimeError(
            "No usable training playlists (no held-out target was ever retrieved). "
            "Increase --n-playlists or --n-candidates."
        )
    print(f'Kept {kept} playlists with >=1 retrieved target.')

    # Seed profile (avg_*) per playlist -- over SEED tracks only (kills circularity).
    df_profiles = (
        pl.DataFrame({'playlist_int_id': seed_pl, 'track_id': seed_tid})
        .join(df_metadata, on='track_id', how='inner')
        .group_by('playlist_int_id')
        .agg([pl.col(tf).mean().alias(af) for tf, af in zip(TRACK_FEATURES, AVG_FEATURES)])
    )

    # Map every candidate track_int_id (across strategies) -> track_id, once.
    all_tints = np.unique(np.concatenate([np.asarray(acc[s]['tint'], dtype=np.int64) for s in strategies]))
    with conn.cursor() as cur:
        cur.execute(
            """SELECT DISTINCT track_int_id, track_id
               FROM interaction_matrix_mapped
               WHERE track_int_id = ANY(%s)""",
            (all_tints.tolist(),),
        )
        map_rows = cur.fetchall()
    df_map = pl.DataFrame(map_rows, schema=['track_int_id', 'track_id'], orient='row').with_columns(
        pl.col('track_int_id').cast(pl.Int64))

    return {s: _build_strategy_df(acc[s], df_map, df_metadata, df_profiles) for s in strategies}


def engineer_features(df):
    """
    Computes the |profile - candidate| cross-features. The profile (avg_*) is
    already attached from fetch_training_data (built over SEED tracks), so this
    just differences it against each candidate's track_* features.
    """
    return df.with_columns([
        (pl.col(af) - pl.col(tf)).abs().alias(diff_col)
        for af, tf, diff_col in zip(AVG_FEATURES, TRACK_FEATURES, DIFF_FEATURES)
    ])


def _fit_and_save(df, out_path, strategy):
    """Fits an XGBRanker on one strategy's feature frame and saves it."""
    processed_df = engineer_features(df)

    # A ranking group needs both a positive and a negative to carry signal.
    label_counts = processed_df.group_by('playlist_int_id').agg(
        pl.col('label').n_unique().alias('n_labels'))
    keep = label_counts.filter(pl.col('n_labels') == 2)['playlist_int_id']
    processed_df = processed_df.filter(pl.col('playlist_int_id').is_in(keep))

    # XGBRanker needs rows grouped contiguously by playlist, with group_counts in the
    # same order as the rows -- so sort first, then derive both from the sorted frame.
    processed_df = processed_df.sort('playlist_int_id')
    group_counts = processed_df.group_by('playlist_int_id', maintain_order=True).len()['len'].to_numpy()

    X = processed_df[FEATURE_COLUMNS].to_pandas()
    y = processed_df['label'].to_numpy()

    n_pos = int((y == 1).sum())
    print(f'[{strategy}] Training XGBRanker on {len(processed_df)} candidates ({n_pos} positive) '
          f'across {len(group_counts)} playlists...')
    ranker = xgb.XGBRanker(
        objective='rank:ndcg',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    ranker.fit(X, y, group=group_counts)

    ranker.save_model(str(out_path))
    print(f'[{strategy}] Saved ranker to {out_path}')
    return ranker


def train_ranker(strategies, out_paths, n_playlists=N_TRAIN_PLAYLISTS, n_candidates=N_CANDIDATES,
                 holdout_frac=HOLDOUT_FRAC, n_neg=N_NEG_PER_PLAYLIST, seed=SEED):
    """
    Trains one XGBRanker per strategy in `strategies` (subset of {'hard','random'})
    and saves each to out_paths[strategy]. The expensive retrieval loop is shared
    across strategies.
    """
    print('Training ranking model(s)...')
    als_model, user_item_matrix = load_als()
    conn = psycopg.connect(DB_URI)
    try:
        dfs = fetch_training_data(
            als_model, user_item_matrix, conn, strategies,
            n_playlists=n_playlists, n_candidates=n_candidates,
            holdout_frac=holdout_frac, n_neg=n_neg, seed=seed,
        )
    finally:
        conn.close()

    models = {}
    for strategy in strategies:
        models[strategy] = _fit_and_save(dfs[strategy], out_paths[strategy], strategy)
    print('Done.')
    return models


def _resolve_out_paths(strategies, out):
    """Maps each strategy to an output path. `--out` only applies to a single strategy."""
    if len(strategies) > 1:
        if out is not None:
            raise SystemExit("--out cannot be used with --negatives both; each strategy gets its own file.")
        return {s: MODELS_DIR / f'xgb_ranker_{s}.json' for s in strategies}
    (strategy,) = tuple(strategies)
    return {strategy: Path(out) if out is not None else MODELS_DIR / f'xgb_ranker_{strategy}.json'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Stage 2 XGBRanker (train-like-serve).')
    parser.add_argument('--negatives', choices=['hard', 'random', 'both'], default='hard',
                        help="Negative-sampling strategy to train. 'both' trains and saves each "
                             "for side-by-side comparison (default: hard).")
    parser.add_argument('--out', type=str, default=None,
                        help="Output model path (single strategy only). Default: "
                             "models/xgb_ranker_<strategy>.json.")
    parser.add_argument('--n-playlists', type=int, default=N_TRAIN_PLAYLISTS,
                        help=f'Playlists sampled for training (default: {N_TRAIN_PLAYLISTS}).')
    parser.add_argument('--n-candidates', type=int, default=N_CANDIDATES,
                        help=f'ALS candidates retrieved per playlist (default: {N_CANDIDATES}).')
    parser.add_argument('--holdout-frac', type=float, default=HOLDOUT_FRAC,
                        help=f'Fraction of each playlist hidden as targets (default: {HOLDOUT_FRAC}).')
    parser.add_argument('--n-neg', type=int, default=N_NEG_PER_PLAYLIST,
                        help=f'Negatives kept per playlist, per strategy (default: {N_NEG_PER_PLAYLIST}).')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'Seed for the seed/target split + random negatives (default: {SEED}).')
    args = parser.parse_args()

    strategies = {'hard', 'random'} if args.negatives == 'both' else {args.negatives}
    out_paths = _resolve_out_paths(strategies, args.out)

    train_ranker(strategies, out_paths, n_playlists=args.n_playlists, n_candidates=args.n_candidates,
                 holdout_frac=args.holdout_frac, n_neg=args.n_neg, seed=args.seed)
