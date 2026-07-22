"""
feature_engineering.py
Builds the Stage 2 ranking features for a retrieved candidate pool.

Three feature families per candidate (one row per candidate):
  - Item features  (track_*) : the candidate's own acoustic metadata.
  - User features  (avg_*)   : the playlist's taste profile, averaged over its
                               tracks. `target` may be an existing
                               playlist_int_id (int) or an arbitrary list of
                               seed track_ids (the cold-start path).
  - Cross features (*_diff)  : |profile - candidate| per acoustic dimension --
                               how far the candidate sits from the playlist's
                               center of taste, which is the actual ranking
                               signal.
Plus `als_score`, the Stage-1 retrieval score.

src/train_ranking.py deliberately mirrors this computation on the training side
(over SEED tracks only, to avoid label leakage) rather than importing it, so
training and serving stay separate.

Run:
    python src/feature_engineering.py
"""

import polars as pl
import psycopg
from contextlib import contextmanager

DB_URI = 'postgresql://postgres:postgres123@localhost:5432/playlist_engine'


@contextmanager
def _connection(conn):
    """Uses `conn` if the caller supplied one (e.g. reusing one connection across
    a whole eval run), otherwise opens and closes a fresh one for this call."""
    if conn is not None:
        yield conn
    else:
        # prepare_threshold=None: never auto-prepare, so `col = ANY($1)` queries are
        # always planned with the actual array values (custom plan) and use the
        # indexes, rather than risking a generic-plan seq scan. (A per-call
        # connection here rarely repeats a query enough to prepare anyway, but a
        # caller may pass in a long-lived shared conn -- see the eval scripts.)
        with psycopg.connect(DB_URI, prepare_threshold=None) as local_conn:
            yield local_conn


def generate_features(target, candidate_track_int_ids, als_scores, conn=None):
    """
    Creates features from track metadata to be used for ranking candidates by the gradient boosting model.

    param target_playlist_int_id: mapped integer index of playlist of interest
    param candidate_track_int_ids: mapped integer indices of track candidates from the ALS model
    param als_scores: Stage-1 ALS retrieval score for each candidate (aligned with
        candidate_track_int_ids). Attached as the `als_score` feature so the ranker
        keeps the collaborative signal instead of reranking on acoustics alone.
    return: dataframe with features of interest (one row per candidate)
    """
    print(f'Extracting features...')

    candidate_ids = [int(i) for i in candidate_track_int_ids]

    # Item-only features: candidate metadata. Dedup the int->id mapping FIRST (<=200
    # rows via idx_track_int), THEN join track_metadata -- otherwise a popular
    # candidate (present in thousands of interaction rows) would explode the join
    # before dedup. track_int_id is carried through so als_score can be joined on it.
    item_query = """
        SELECT
            map.track_int_id,
            t.track_id,
            t.danceability AS track_danceability,
            t.tempo AS track_tempo,
            t.energy AS track_energy,
            t.acousticness AS track_acousticness,
            t.loudness AS track_loudness,
            t.valence AS track_valence
        FROM (
            SELECT DISTINCT track_int_id, track_id
            FROM interaction_matrix_mapped
            WHERE track_int_id = ANY(%s)
        ) map
        JOIN track_metadata t ON map.track_id = t.track_id
    """

    # User-only features: metadata averages across playlist
    if isinstance(target, int):
        user_query = """
            SELECT
                i.playlist_id,
                AVG(m.danceability) AS avg_danceability,
                AVG(m.tempo) AS avg_tempo,
                AVG(m.energy) AS avg_energy,
                AVG(m.acousticness) AS avg_acousticness,
                AVG(m.loudness) AS avg_loudness,
                AVG(m.valence) AS avg_valence,
                COUNT(i.track_id) AS playlist_length
            FROM interaction_matrix i
            JOIN track_metadata m ON i.track_id = m.track_id
            WHERE i.playlist_id = (
                SELECT playlist_id
                FROM interaction_matrix_mapped
                WHERE playlist_int_id = %s
                LIMIT 1
            )
            GROUP BY i.playlist_id
        """
        user_params = (target,)

    elif isinstance(target, list):
        user_query = """
            SELECT
                AVG(danceability) AS avg_danceability,
                AVG(tempo) AS avg_tempo,
                AVG(energy) AS avg_energy,
                AVG(acousticness) AS avg_acousticness,
                AVG(loudness) AS avg_loudness,
                AVG(valence) AS avg_valence,
                COUNT(*) AS playlist_length
            FROM track_metadata
            WHERE track_id = ANY(%s)
        """
        user_params = (target,)

    else:
        raise TypeError(f"target must be an int (playlist_int_id) or list (seed track_ids), got {type(target)}")

    with _connection(conn) as active_conn, active_conn.cursor() as cur:
        cur.execute(item_query, (candidate_ids,))
        item_rows = cur.fetchall()
        item_columns = [desc.name for desc in cur.description]

        cur.execute(user_query, user_params)
        user_rows = cur.fetchall()
        user_columns = [desc.name for desc in cur.description]

    df_items = pl.DataFrame(item_rows, schema=item_columns, orient='row')
    df_user = pl.DataFrame(user_rows, schema=user_columns, orient='row')

    # Attach the ALS retrieval score to each candidate BY track_int_id (never
    # positionally -- df_items comes back in DB order and drops metadata-less
    # candidates, so positional alignment would be wrong).
    df_scores = pl.DataFrame({
        'track_int_id': [int(i) for i in candidate_track_int_ids],
        'als_score': [float(s) for s in als_scores],
    }).with_columns(pl.col('track_int_id').cast(pl.Int64))
    df_items = df_items.with_columns(pl.col('track_int_id').cast(pl.Int64)).join(
        df_scores, on='track_int_id', how='left')

    # Cross features
    print('Create cross features...')

    df_features = df_items.join(df_user, how='cross')
    # Calculate absolute differences
    df_features = df_features.with_columns([
        (pl.col('avg_danceability') - pl.col('track_danceability')).abs().alias('danceability_diff'),
        (pl.col('avg_tempo') - pl.col('track_tempo')).abs().alias('tempo_diff'),
        (pl.col('avg_energy') - pl.col('track_energy')).abs().alias('energy_diff'),
        (pl.col('avg_acousticness') - pl.col('track_acousticness')).abs().alias('acousticness_diff'),
        (pl.col('avg_loudness') - pl.col('track_loudness')).abs().alias('loudness_diff'),
        (pl.col('avg_valence') - pl.col('track_valence')).abs().alias('valence_diff')
    ])

    return df_features




if __name__ == '__main__':
    # Test
    test_playlist_id = ['0UaMYEvWZi0ZqiDOoHU3YI','6I9VzXrHxO9rA9A5euc8Ak','0WqIKmW4BTrj3eJFmnCKMv','1AWQoqb9bSvzTjaLralEkT']
    mock_candidates = [101, 205, 310, 450, 512]
    mock_scores = [0.9, 0.8, 0.7, 0.6, 0.5]

    df_ranking_dataset = generate_features(test_playlist_id, mock_candidates, mock_scores)

    print('\nSuccessfully compiled ranking features!')
    print(df_ranking_dataset.head())