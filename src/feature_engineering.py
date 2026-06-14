import polars as pl

def generate_features(target_playlist_int_id, candidate_track_int_ids):
    """
    Creates features from track metadata to be used for ranking candidates by the gradient boosting model.

    param target_playlist_int_id: mapped integer index of playlist of interest
    param candidate_track_int_ids: mapped integer indices of track candidates from the ALS model
    return: dataframe with features of interest
    """
    print(f'Extracting features for playlist int ID {target_playlist_int_id}...')

    DB_URI = 'postgresql://postgres:postgres123@localhost:5432/playlist_engine'

    id_list_str = ', '.join(map(str, candidate_track_int_ids))

    # Item-only features: candidate metadata
    item_query = f"""
        SELECT 
            track_id,
            danceability AS track_danceability,
            tempo AS track_tempo,
            energy AS track_energy,
            acousticness AS track_acousticness, 
            loudness AS track_loudness, 
            valence AS track_valence
        FROM track_metadata
        WHERE track_id IN (
            SELECT DISTINCT track_id
            FROM interaction_matrix_mapped
            WHERE track_int_id IN ({id_list_str})
        )
    """
    df_items = pl.read_database_uri(query=item_query, uri=DB_URI, engine='connectorx')

    # User-only features: metadata averages across playlist
    user_query = f"""
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
            WHERE playlist_int_id = {target_playlist_int_id}
            LIMIT 1
        )
        GROUP BY i.playlist_id
    """
    df_user = pl.read_database_uri(query=user_query, uri=DB_URI, engine='connectorx')

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
    test_playlist_id = 0
    mock_candidates = [101, 205, 310, 450, 512]

    df_ranking_dataset = generate_features(test_playlist_id, mock_candidates)

    print('\nSuccessfully compiled ranking features!')
    print(df_ranking_dataset.head())