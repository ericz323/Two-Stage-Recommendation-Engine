import polars as pl
import xgboost as xgb
from implicit.als import AlternatingLeastSquares
from scipy.sparse import load_npz

from feature_engineering import generate_features

print('Loading models into memory...')
als_model = AlternatingLeastSquares().load('als_model.npz')
user_item_matrix = load_npz('user_item_matrix.npz')

xgb_model = xgb.XGBRanker()
xgb_model.load_model('xgb_ranker.json')

def generate_recommendations(target_playlist_int_id, als_model, user_item_matrix, xgb_model):
    """
    Generates top 20 song recommendations for a given playlist.

    param target_playlist_int_id: mapped integer index for playlist of interest
    param als_model: trained alternating least squares model
    param user_item_matrix: compressed sparse matrix of playlist-track interactions
    param xgb_model: trained gradient boosting ranking model
    return: dataframe of recommendations for a given playlist
    """
    print(f'Generating recommendations for playlist {target_playlist_int_id}...')

    # 1. Retrieval
    print('[1/4] Running vector retrieval...')
    candidate_integers, als_scores = als_model.recommend(
        userid = target_playlist_int_id,
        user_items = user_item_matrix[target_playlist_int_id],
        N=200
    )

    # 2. Feature engineering
    print('[2/4] Engineering cross-features...')
    df_features = generate_features(target_playlist_int_id, candidate_integers)

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
    df_top_20 = df_ranked.sort('xgb_score', descending=True).head(20)

    # 4. Presentation
    print('[4/4] Generating output...')
    DB_URI = 'postgresql://postgres:postgres123@localhost:5432/playlist_engine'

    final_track_ids = ', '.join(f"'{tid}'" for tid in df_top_20['track_id'].to_list())

    name_query = f"""
        SELECT track_id, track_name, artist_name
        FROM track_metadata
        WHERE track_id IN ({final_track_ids})
    """
    df_names = pl.read_database_uri(query=name_query, uri=DB_URI, engine='connectorx')
    df_presentation = df_top_20.join(df_names, on='track_id', how='left').sort('xgb_score', descending=True)

    print('\n=======================================================')
    print('YOUR RECOMMENDATIONS')
    print('=======================================================')
    print(df_presentation.select(['artist_name', 'track_name', 'xgb_score']))
    print('=======================================================\n')

    return df_presentation


if __name__ == '__main__':
    # Test with Playlist ID 0
    test_playlist_int_id = 0

    generate_recommendations(test_playlist_int_id, als_model, user_item_matrix, xgb_model)