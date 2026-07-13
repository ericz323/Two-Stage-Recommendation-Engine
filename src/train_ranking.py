"""
train_ranking.py
Trains the Stage 2 XGBRanker that reranks the ALS retrieval candidates.

Trained the SAME way it is served ("train like you serve"). For each playlist we
hide a fraction of its tracks (the TARGET), retrieve candidates from the
remaining SEED tracks via the exact cold-start path inference uses
(recalculate_user), and label each retrieved candidate 1 if it is a hidden
target track and 0 otherwise. So:

  - Negatives are HARD negatives -- tracks ALS retrieved from the seed that are
    not actually held-out targets -- the same pool the ranker reorders at serving.
  - The taste profile (avg_*) is computed over the SEED tracks only, never the
    labeled targets, so the *_diff cross-features mean the same thing at train and
    serve time (no "positives define their own centroid" circularity).
  - als_score is the recalculate_user retrieval score -- identical in origin to
    the als_score feature computed at inference -- so there is no train/serve skew
    in that feature either.

Because positives are only the retrieved held-out targets, most playlists yield
few (or zero) positives; playlists with none are dropped. Increase --n-playlists
or --n-candidates to get more positive coverage.
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
N_NEG_PER_PLAYLIST = 50    # hard negatives kept per playlist (highest-scored)
SEED = 42                  # seed for the per-playlist seed/target split

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


def fetch_training_data(als_model, user_item_matrix, conn, n_playlists=N_TRAIN_PLAYLISTS,
                        n_candidates=N_CANDIDATES, holdout_frac=HOLDOUT_FRAC,
                        n_neg=N_NEG_PER_PLAYLIST, seed=SEED):
    """
    Builds the labeled training set by mirroring the serving path per playlist:
    split tracks into seed/target, retrieve candidates from the seed via
    recalculate_user, label a candidate 1 iff it is a held-out target.

    return: polars df with playlist_int_id, label, als_score, the six track_*
            acoustic features, and the six avg_* seed-profile features.
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

    print(f'Retrieving candidates from seed tracks (recalculate_user) for {len(playlist_tracks)} playlists...')
    n_items = user_item_matrix.shape[1]
    seed_pl, seed_tid = [], []                      # (playlist, seed track_id) -> seed profile
    cand_pl, cand_tint, cand_label, cand_score = [], [], [], []
    kept = 0
    for pid, tracks in playlist_tracks.items():
        if len(tracks) < 2:
            continue  # need >=1 seed and >=1 target
        seed_pairs, target_pairs = _split_holdout(tracks, holdout_frac, seed=_stable_seed(pid, seed))
        if not seed_pairs:
            continue
        seed_tints = [t[0] for t in seed_pairs]
        target_tints = {int(t[0]) for t in target_pairs}

        # Build the seed user vector exactly as get_live_recommendations does, then
        # retrieve via recalculate_user -- the identical Stage-1 path used at serving.
        user_row = csr_matrix(
            (np.ones(len(seed_tints)), (np.zeros(len(seed_tints)), np.asarray(seed_tints))),
            shape=(1, n_items),
        )
        ids, scores = als_model.recommend(
            0, user_row, N=n_candidates, recalculate_user=True, filter_already_liked_items=True)
        ids = np.asarray(ids).reshape(-1)
        scores = np.asarray(scores).reshape(-1)
        valid = ids >= 0
        ids, scores = ids[valid], scores[valid]
        if len(ids) == 0:
            continue

        labels = np.fromiter((1 if int(t) in target_tints else 0 for t in ids), dtype=int, count=len(ids))
        if labels.sum() == 0:
            continue  # no held-out target was retrieved -> no positive -> unusable group

        # Keep all positives + the hardest n_neg negatives (ids come back score-desc,
        # so the first negatives are the highest-scored = most confusable).
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0][:n_neg]
        for i in np.concatenate([pos_idx, neg_idx]):
            cand_pl.append(pid)
            cand_tint.append(int(ids[i]))
            cand_label.append(int(labels[i]))
            cand_score.append(float(scores[i]))
        for _tint, tid in seed_pairs:
            seed_pl.append(pid)
            seed_tid.append(tid)
        kept += 1

    if kept == 0:
        raise RuntimeError(
            "No usable training playlists (no held-out target was ever retrieved). "
            "Increase --n-playlists or --n-candidates."
        )
    print(f'Kept {kept} playlists with >=1 retrieved target; {len(cand_pl)} candidate rows.')

    # Seed profile (avg_*) per playlist -- over SEED tracks only (kills circularity).
    df_profiles = (
        pl.DataFrame({'playlist_int_id': seed_pl, 'track_id': seed_tid})
        .join(df_metadata, on='track_id', how='inner')
        .group_by('playlist_int_id')
        .agg([pl.col(tf).mean().alias(af) for tf, af in zip(TRACK_FEATURES, AVG_FEATURES)])
    )

    # Map candidate track_int_id -> track_id, then attach candidate metadata + profile.
    with conn.cursor() as cur:
        cur.execute(
            """SELECT DISTINCT track_int_id, track_id
               FROM interaction_matrix_mapped
               WHERE track_int_id = ANY(%s)""",
            (np.unique(np.asarray(cand_tint)).tolist(),),
        )
        map_rows = cur.fetchall()
    df_map = pl.DataFrame(map_rows, schema=['track_int_id', 'track_id'], orient='row').with_columns(
        pl.col('track_int_id').cast(pl.Int64))

    df = (
        pl.DataFrame({
            'playlist_int_id': cand_pl,
            'track_int_id': cand_tint,
            'label': cand_label,
            'als_score': cand_score,
        })
        .with_columns(pl.col('track_int_id').cast(pl.Int64), pl.col('als_score').cast(pl.Float64))
        .join(df_map, on='track_int_id', how='inner')
        .drop('track_int_id')
        .join(df_metadata, on='track_id', how='inner')       # candidate track_* features
        .join(df_profiles, on='playlist_int_id', how='inner')  # avg_* seed profile
    )
    return df


def engineer_features(df):
    """
    Computes the |profile - candidate| cross-features. The profile (avg_*) is
    already attached from fetch_training_data (built over SEED tracks), so this
    just differences it against each candidate's track_* features.
    """
    print('Calculating profile alignment features...')
    return df.with_columns([
        (pl.col(af) - pl.col(tf)).abs().alias(df_col)
        for af, tf, df_col in zip(AVG_FEATURES, TRACK_FEATURES, DIFF_FEATURES)
    ])


def train_ranker(n_playlists=N_TRAIN_PLAYLISTS, n_candidates=N_CANDIDATES,
                 holdout_frac=HOLDOUT_FRAC, n_neg=N_NEG_PER_PLAYLIST, seed=SEED):
    """
    Trains the gradient boosting ranking model on the train-like-serve data.
    return: trained model
    """
    print('Training ranking model...')

    als_model, user_item_matrix = load_als()
    conn = psycopg.connect(DB_URI)
    try:
        raw_df = fetch_training_data(
            als_model, user_item_matrix, conn,
            n_playlists=n_playlists, n_candidates=n_candidates,
            holdout_frac=holdout_frac, n_neg=n_neg, seed=seed,
        )
    finally:
        conn.close()

    processed_df = engineer_features(raw_df)

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
    print(f'Training XGBRanker on {len(processed_df)} candidates ({n_pos} positive) '
          f'across {len(group_counts)} playlists...')
    ranker = xgb.XGBRanker(
        objective='rank:ndcg',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    ranker.fit(X, y, group=group_counts)
    print('XGBRanker training complete!')

    print('Saving model...')
    ranker.save_model(str(MODELS_DIR / 'xgb_ranker.json'))
    print('XGBRanker model saved!')

    return ranker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Stage 2 XGBRanker (train-like-serve).')
    parser.add_argument('--n-playlists', type=int, default=N_TRAIN_PLAYLISTS,
                        help=f'Playlists sampled for training (default: {N_TRAIN_PLAYLISTS}).')
    parser.add_argument('--n-candidates', type=int, default=N_CANDIDATES,
                        help=f'ALS candidates retrieved per playlist (default: {N_CANDIDATES}).')
    parser.add_argument('--holdout-frac', type=float, default=HOLDOUT_FRAC,
                        help=f'Fraction of each playlist hidden as targets (default: {HOLDOUT_FRAC}).')
    parser.add_argument('--n-neg', type=int, default=N_NEG_PER_PLAYLIST,
                        help=f'Hard negatives kept per playlist (default: {N_NEG_PER_PLAYLIST}).')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'Seed for the per-playlist seed/target split (default: {SEED}).')
    args = parser.parse_args()

    train_ranker(n_playlists=args.n_playlists, n_candidates=args.n_candidates,
                 holdout_frac=args.holdout_frac, n_neg=args.n_neg, seed=args.seed)
