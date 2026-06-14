import polars as pl
import xgboost as xgb
import numpy as np

DB_URI = "postgresql://postgres:postgres123@localhost:5432/playlist_engine"

def fetch_training_data():
    """
    Processes training data for the gradient boosting model.

    Retrieves a certain amount of 'positive' interactions from the database. Then randomly generates playlist-track
    pairings as 'negative' interactions.

    return: dataframe of interaction data labeled '1' for positive and '0' for negative
    """
    print('Fetching positive interactions...')
    query_pos = """SELECT playlist_id, track_id FROM interaction_matrix LIMIT 500000"""
    df_pos = pl.read_database_uri(query=query_pos, uri=DB_URI, engine='connectorx')
    df_pos = df_pos.with_columns(pl.lit(1).alias('label'))

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
    all_tracks = df_metadata['track_id'].to_numpy()

    print('Generating negative interactions...')
    num_negatives = len(df_pos) * 4
    unique_playlists = df_pos['playlist_id'].unique().to_numpy()

    # Generate random combinations of playlists and tracks
    random_playlists = np.random.choice(unique_playlists, num_negatives, replace=True)
    random_tracks = np.random.choice(all_tracks, num_negatives, replace=True)

    df_neg_raw = pl.DataFrame({
        'playlist_id': random_playlists,
        'track_id': random_tracks
    })

    # Drop pairs that actually exist in the playlists
    df_neg = df_neg_raw.join(df_pos, on=['playlist_id', 'track_id'], how='anti')
    df_neg = df_neg.with_columns(pl.lit(0).alias('label'))

    df_pos_neg = pl.concat([df_pos, df_neg])

    print('Attaching acoustic features to the training data...')
    df_final = df_pos_neg.join(df_metadata, on='track_id', how='inner')
    df_final = df_final.sort('playlist_id')

    return df_final


def engineer_features(df):
    """
    Creates features for model training.
    param df: training interactions data
    return: dataframe of training data with features of interest
    """
    print('Calculating profile alignment features...')

    df_playlist_profiles = df.group_by('playlist_id').agg([
        pl.col('track_danceability').mean().alias('avg_danceability'),
        pl.col('track_tempo').mean().alias('avg_tempo'),
        pl.col('track_energy').mean().alias('avg_energy'),
        pl.col('track_acousticness').mean().alias('avg_acousticness'),
        pl.col('track_loudness').mean().alias('avg_loudness'),
        pl.col('track_valence').mean().alias('avg_valence')
    ])

    df_features = df.join(df_playlist_profiles, on='playlist_id', how='left')

    df_features = df_features.with_columns([
        (pl.col('avg_danceability') - pl.col('track_danceability')).abs().alias('danceability_diff'),
        (pl.col('avg_tempo') - pl.col('track_tempo')).abs().alias('tempo_diff'),
        (pl.col('avg_energy') - pl.col('track_energy')).abs().alias('energy_diff'),
        (pl.col('avg_acousticness') - pl.col('track_acousticness')).abs().alias('acousticness_diff'),
        (pl.col('avg_loudness') - pl.col('track_loudness')).abs().alias('loudness_diff'),
        (pl.col('avg_valence') - pl.col('track_valence')).abs().alias('valence_diff')
    ])

    return df_features


def train_ranker():
    """
    Trains the gradient boosting ranking model.
    return: trained model
    """
    print('Training ranking model...')

    raw_df = fetch_training_data()
    processed_df = engineer_features(raw_df)

    group_counts = processed_df.group_by('playlist_id', maintain_order=True).len()['len'].to_numpy()

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
    X = processed_df[feature_columns].to_pandas()
    y = processed_df['label'].to_numpy()

    print(f'Training XGBRanker on {len(processed_df)} candidates across {len(group_counts)} playlists...')
    ranker = xgb.XGBRanker(
        objective='rank:ndcg',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    ranker.fit(X, y, group=group_counts)
    print('XGBRanker training complete!')

    # Save XGBoost model
    print('Saving model...')
    ranker.save_model('xgb_ranker.json')
    print('XGBRanker model saved!')

    return ranker


if __name__ == '__main__':
    model = train_ranker()