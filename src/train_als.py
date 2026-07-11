import polars as pl
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from implicit.als import AlternatingLeastSquares
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

def build_csr_matrix():
    """
    Creates a sparse user-item matrix whose rows are playlists and columns are tracks.

    return: csr_matrix user-item matrix
    """
    print('Connecting to PostgreSQL database...')
    DB_URI = "postgresql://postgres:postgres123@localhost:5432/playlist_engine"

    print('Extracting mapped integer data...')
    query = """SELECT playlist_int_id::INT, track_int_id::INT FROM interaction_matrix_mapped;"""
    df_interactions = pl.read_database_uri(query, uri=DB_URI, engine='connectorx')

    playlist_indices = df_interactions['playlist_int_id'].to_numpy()
    track_indices = df_interactions['track_int_id'].to_numpy()

    # Set baseline confidence of songs already in playlists to 1 for that interaction
    confidence_score = np.ones(len(df_interactions), dtype=np.float32)

    print('Compressing into a sparse matrix...')
    num_playlists = playlist_indices.max() + 1
    num_tracks = track_indices.max() + 1
    sparse_user_item_matrix = csr_matrix((confidence_score, (playlist_indices, track_indices)),
                                         shape=(num_playlists, num_tracks))

    return sparse_user_item_matrix


def train_als_model(user_item_matrix):
    """
    Trains an Alternating Least Squares factorization model with a given user-item matrix.

    param user_item_matrix: compressed sparse matrix of playlist-track interactions
    """
    print('Training ALS Matrix Factorization model...')
    # 50 latent features
    model = AlternatingLeastSquares(factors=50, iterations=15, regularization=0.01)
    model.fit(user_item_matrix)
    print('Training complete!')

    print('Saving model...')
    model.save(str(MODELS_DIR / 'als_model.npz'))
    save_npz(str(MODELS_DIR / 'user_item_matrix.npz'), user_item_matrix)
    print('ALS model saved!')

    # # Test retrieval
    # target_index=0
    # candidates, scores = model.recommend(
    #     target_index,
    #     user_item_matrix[target_index],
    #     N=200
    # )
    #
    # print(f'Successfully retrieved {len(candidates)} recommendations for playlist {target_index}.')


if __name__ == '__main__':
    csr_matrix = build_csr_matrix()
    train_als_model(csr_matrix)