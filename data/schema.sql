DROP TABLE IF EXISTS interaction_matrix;
DROP TABLE IF EXISTS track_metadata;

CREATE TABLE track_metadata (
    track_id VARCHAR(255) PRIMARY KEY,
    artist_name TEXT,
    track_name TEXT,
    danceability REAL,
    tempo REAL,
    energy REAL,
    acousticness REAL,
    loudness REAL,
    valence REAL
);

CREATE TABLE interaction_matrix (
    interaction_id SERIAL PRIMARY KEY,
    playlist_id VARCHAR(255),
    track_id VARCHAR(255) REFERENCES track_metadata(track_id)
);

CREATE INDEX idx_interaction_playlist ON interaction_matrix(playlist_id);
CREATE INDEX idx_interaction_track ON interaction_matrix(track_id);
CREATE INDEX idx_meta_track_id ON track_metadata(track_id);