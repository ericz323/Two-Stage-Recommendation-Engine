import argparse
import polars as pl
import numpy as np
import psycopg
import xgboost as xgb
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, load_npz
from pathlib import Path

from src.feature_engineering import generate_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DB_URI = 'postgresql://postgres:postgres123@localhost:5432/playlist_engine'

def get_live_recommendations(track_ids, als_model, user_item_matrix, num_recs=20):
    track_int_query = """
        SELECT track_int_id
        FROM interaction_matrix_mapped
        WHERE track_id = ANY(%s)
    """
    with psycopg.connect(DB_URI) as conn, conn.cursor() as cur:
        cur.execute(track_int_query, (list(track_ids),))
        track_ints = [row[0] for row in cur.fetchall()]

    data = np.ones(len(track_ints))
    rows = np.zeros(len(track_ints))
    cols = np.array(track_ints)
    user_items = csr_matrix((data, (rows, cols)), shape=(1, user_item_matrix.shape[1]))

    candidate_integers, als_scores = als_model.recommend(
        userid=0,
        user_items=user_items,
        N=num_recs,
        recalculate_user=True
    )

    return candidate_integers, als_scores


def get_als_candidates(target, als_model, user_item_matrix, n=200):
    """
    Runs Stage 1 (ALS retrieval) only, for either an existing playlist_int_id
    (int) or an arbitrary list of seed track_ids. Shared by generate_recommendations
    and get_als_only_recommendations so both paths retrieve candidates identically.

    return: (candidate_integers, als_scores), already ranked by ALS score descending
    """
    if isinstance(target, int):
        candidate_integers, als_scores = als_model.recommend(
            userid=target,
            user_items=user_item_matrix[target],
            N=n
        )
    elif isinstance(target, list):
        candidate_integers, als_scores = get_live_recommendations(target, als_model, user_item_matrix, num_recs=n)
    else:
        raise TypeError(f"target must be an int (playlist_int_id) or list (seed track_ids), got {type(target)}")

    return candidate_integers, als_scores


def map_track_ints_to_ids(candidate_integers):
    """
    Maps ALS's internal track_int_id candidates back to track_id strings,
    preserving the input (ALS-ranked) order.
    """
    query = """
        SELECT DISTINCT track_int_id, track_id
        FROM interaction_matrix_mapped
        WHERE track_int_id = ANY(%s)
    """
    candidate_ids = [int(i) for i in candidate_integers]
    with psycopg.connect(DB_URI) as conn, conn.cursor() as cur:
        cur.execute(query, (candidate_ids,))
        int_to_id = {row[0]: row[1] for row in cur.fetchall()}

    return [int_to_id[int(i)] for i in candidate_integers if int(i) in int_to_id]


def get_als_only_recommendations(target, als_model, user_item_matrix, k=20):
    """
    Stage 1 (ALS retrieval) ONLY -- skips XGBRanker reranking entirely.
    Isolates whether Stage 2 reranking is adding value over raw ALS retrieval.

    param target: playlist_int_id (int) or list of seed track_ids
    return: list of track_id strings, ranked by ALS score, length k
    """
    candidate_integers, _als_scores = get_als_candidates(target, als_model, user_item_matrix, n=k)
    return map_track_ints_to_ids(candidate_integers)


def generate_recommendations(target, als_model, user_item_matrix, xgb_model, k=20):
    """
    Generates top-k song recommendations for a given playlist.

    param target_playlist_int_id: mapped integer index for playlist of interest
    param als_model: trained alternating least squares model
    param user_item_matrix: compressed sparse matrix of playlist-track interactions
    param xgb_model: trained gradient boosting ranking model
    param k: number of final recommendations to return
    return: dataframe of recommendations for a given playlist
    """


    print(f'Generating recommendations...')

    # 1. Retrieval
    print('[1/4] Creating candidates...')
    candidate_integers, als_scores = get_als_candidates(target, als_model, user_item_matrix, n=200)

    # 2. Feature engineering
    print('[2/4] Engineering cross-features...')
    df_features = generate_features(target, candidate_integers)

    # 3. Ranking
    print('[3/4] Ranking candidates...')
    feature_columns = [
        'track_danceability',
        'track_tempo',
        'track_energy',
        'track_acousticness',
        'track_loudness',
        'track_valence',
        'danceability_diff',
        'tempo_diff',
        'energy_diff',
        'acousticness_diff',
        'loudness_diff',
        'valence_diff'
    ]
    X_inf = df_features[feature_columns].to_pandas()

    xgb_scores = xgb_model.predict(X_inf)

    df_ranked = df_features.with_columns(pl.Series('xgb_score', xgb_scores))
    df_top_20 = df_ranked.sort('xgb_score', descending=True).head(k)

    return df_top_20


def report(top_20):
    print("Generating output...")

    name_query = """
        SELECT track_id, track_name, artist_name
        FROM track_metadata
        WHERE track_id = ANY(%s)
    """
    with psycopg.connect(DB_URI) as conn, conn.cursor() as cur:
        cur.execute(name_query, (top_20['track_id'].to_list(),))
        rows = cur.fetchall()
    df_names = pl.DataFrame(rows, schema=['track_id', 'track_name', 'artist_name'], orient='row')
    df_presentation = top_20.join(df_names, on='track_id', how='left').sort('xgb_score', descending=True)

    print('\n=======================================================')
    print('YOUR RECOMMENDATIONS')
    print('=======================================================')
    print(df_presentation.select(['artist_name', 'track_name', 'xgb_score']))
    print('=======================================================\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate track recommendations for a playlist.')
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        '--playlist-id',
        type=int,
        help='Mapped integer playlist_int_id of an existing playlist.'
    )
    target_group.add_argument(
        '--track-ids',
        nargs='+',
        help='One or more seed track IDs (space-separated) to recommend from.'
    )
    parser.add_argument(
        '-k', '--num-recs',
        type=int,
        default=20,
        help='Number of recommendations to return (default: 20).'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    target = args.playlist_id if args.playlist_id is not None else args.track_ids

    print('Loading models into memory...')
    als_model = AlternatingLeastSquares().load(str(MODELS_DIR / 'als_model.npz'))
    user_item_matrix = load_npz(str(MODELS_DIR / 'user_item_matrix.npz'))
    xgb_model = xgb.XGBRanker()
    xgb_model.load_model(str(MODELS_DIR / 'xgb_ranker.json'))

    top_20 = generate_recommendations(target, als_model, user_item_matrix, xgb_model, k=args.num_recs)
    report(top_20)
