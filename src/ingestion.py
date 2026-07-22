"""
ingestion.py
Builds the PostgreSQL foundation the whole engine reads from.

Three stages, in order:
  1. process_metadata()        -- skims every MPD slice for unique track_ids and
                                  left-joins the acoustic features from the
                                  Kaggle SQLite archive -> `track_metadata`.
                                  This is the shift-left ETL: features are merged
                                  in once here, so inference never calls an
                                  audio-features API.
  2. process_interactions()    -- streams (playlist_id, track_id) pairs from the
                                  MPD one slice at a time (to cap memory) into
                                  `interaction_matrix`.
  3. create_integer_mappings() -- adds `interaction_matrix_mapped`, which
                                  DENSE_RANKs playlist_id/track_id into the
                                  contiguous 0-based integers ALS requires, and
                                  indexes both. The aggregation runs inside
                                  Postgres to keep Python's footprint small.

NOTE: JSON_FOLDER and the SQLite URI are relative paths  -- run this from `src/`

Run:
    python ingestion.py
"""

import polars as pl
import json
import glob
import psycopg
from sqlalchemy import create_engine

PG_URI = 'postgresql://postgres:postgres123@localhost:5432/playlist_engine'
# Directory for the MPD folder
JSON_FOLDER = '../data/raw/spotify_mpd'


def process_metadata():
    """
    Processes track metadata and uploads it to the PostgreSQL database.

    Joins all unique tracks from the MPD to a song features dataset from Kaggle.
    """
    print('Processing metadata...')
    file_paths = glob.glob(f'{JSON_FOLDER}/mpd.slice.*.json')

    # Skim MPD for all unique tracks
    unique_tracks = set()
    for i, file_path in enumerate(file_paths):
        if i % 100 == 0:
            print(f'Skimming file {i}/1000...')

        with open(file_path, 'r') as f:
            data = json.load(f)
            for playlist in data['playlists']:
                for track in playlist['tracks']:
                    clean_id = track['track_uri'].split(':')[-1]
                    unique_tracks.add(clean_id)

    print(f'Found {len(unique_tracks)} unique tracks across all playlists.')

    # Convert the set to a DataFrame
    df_all_tracks = pl.DataFrame({'track_id': list(unique_tracks)})

    # Join with metadata from Kaggle dataset (sqlite)
    SQLITE_URI = "sqlite://../data/raw/extracted.sqlite"
    query = """
        SELECT
            track_uri,
            artist_name,
            track_name,
            danceability,
            tempo,
            energy,
            acousticness,
            loudness,
            valence
        FROM extracted;
    """
    df_features = pl.read_database_uri(query, uri=SQLITE_URI, engine='connectorx')

    print(f'Extracted {len(df_features)} tracks from SQLite.')
    df_features = df_features.with_columns(
        pl.col('track_uri').str.replace('spotify:track:', '').alias('track_id')
    ).drop('track_uri')

    df_metadata = df_all_tracks.join(df_features, on='track_id', how='left')

    # Push catalog to Database
    print('Uploading complete metadata catalog to PostgreSQL...')
    df_metadata.write_database(
        table_name='track_metadata',
        connection=PG_URI,
        if_table_exists='append',
        engine='sqlalchemy'
    )


def process_interactions():
    """
    Processes track interactions from the MPD and uploads them to the PostgreSQL database.

    We select only the playlist ID and track ID. The MPD is processed one slice at a time to conserve memory.
    """
    print('Processing interactions...')
    file_paths = glob.glob(f'{JSON_FOLDER}/mpd.slice.*.json')

    # Process and upload one file at a time to save memory
    for i, file_path in enumerate(file_paths):
        interactions = []

        with open(file_path, 'r') as f:
            data = json.load(f)
            for playlist in data['playlists']:
                pid = playlist['pid']
                for track in playlist['tracks']:
                    clean_id = track['track_uri'].split(':')[-1]
                    interactions.append((pid, clean_id))

        # Build DataFrame for just this ONE slice
        df_batch = pl.DataFrame(interactions, schema=['playlist_id', 'track_id'], orient='row')

        # Append directly to database
        df_batch.write_database(
            table_name='interaction_matrix',
            connection=PG_URI,
            if_table_exists='append',
            engine='sqlalchemy'
        )

        if i % 50 == 0:
            print(f'Successfully piped {i}/1000 slices to database...')

    print('Successfully ingested all interactions.')


def create_integer_mappings():
    """
    Creates a new table that maps each of the playlist IDs and track IDs from interaction_matrix to contiguous integers
    starting at 0.

    The ALS model requires indices to be contiguous integers starting at 0.
    """
    print('Generating integer mappings...')

    with psycopg.connect(PG_URI) as conn:
        with conn.cursor() as cur:
            cur.execute("""DROP TABLE IF EXISTS interaction_matrix_mapped;""")

            cur.execute("""
                CREATE TABLE interaction_matrix_mapped AS
                    SELECT
                        playlist_id,
                        (DENSE_RANK() OVER (ORDER BY playlist_id)) - 1 AS playlist_int_id,
                        track_id,
                        (DENSE_RANK() OVER (ORDER BY track_id)) - 1 AS track_int_id
                    FROM interaction_matrix;
            """)

            # Create indices
            print('Building indices...')
            cur.execute("""CREATE INDEX idx_playlist_int ON interaction_matrix_mapped(playlist_int_id);""")
            cur.execute("""CREATE INDEX idx_track_int ON interaction_matrix_mapped(track_int_id);""")

            conn.commit()

    print('Integer mappings created!')


if __name__ == '__main__':
    engine = create_engine(PG_URI)
    process_metadata()
    process_interactions()
    create_integer_mappings()