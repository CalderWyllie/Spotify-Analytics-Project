import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# Loading and Cleaning Data

df = pd.read_csv("Popular_Spotify_Songs.csv", encoding="latin1")


df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
      .str.replace("(", "", regex=False)
      .str.replace(")", "", regex=False)
)


comma_cols = [
    "streams",
    "in_spotify_playlists",
    "in_spotify_charts",
    "in_apple_playlists",
    "in_apple_charts",
    "in_deezer_playlists",
    "in_deezer_charts",
    "in_shazam_charts"
]

for col in comma_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")


percent_cols = [
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%"
]

for col in percent_cols:
    df[col] = df[col] / 100.0


df = df.drop_duplicates()


numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Feature Engineering
df["mood_score"] = df["valence_%"] * df["danceability_%"]

df["intensity_score"] = (
    0.6 * df["energy_%"] +
    0.3 * df["danceability_%"] +
    0.1 * (df["bpm"] / df["bpm"].max())
)

# Popularity Prediction Model


y = df["streams"]


feature_cols = [
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
    "bpm",
    "artist_count",
    "in_spotify_playlists",
    "in_spotify_charts",
    "in_apple_playlists",
    "in_apple_charts",
    "in_deezer_playlists",
    "in_deezer_charts",
    "in_shazam_charts",
    "released_year",
    "released_month",
    "released_day",
    "mood_score",
    "intensity_score",
]

X = df[feature_cols]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)

importances = pd.Series(rf_model.feature_importances_, index=feature_cols)


# Content Based Reccomendation

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Features to use for similarity
rec_features = [
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
    "bpm",
    "mood_score",
    "intensity_score",
]


scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[rec_features])

# Cosine similarity matrix
similarity_matrix = cosine_similarity(scaled_features)


def recommend(song_name, n=5):
    
    song_name_lower = song_name.lower()

    
    matches = df[df["track_name"].str.lower() == song_name_lower]

    if matches.empty:
        print(f"Song '{song_name}' not found.")
        return

    
    idx = matches.index[0]

    
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  
    top_indices = [i for i, _ in sim_scores[1:n+1]]

    if "artists_name" in df.columns:
        artist_col = "artists_name"
    else:
        artist_col = "artist(s)_name"

    result = df.loc[top_indices, ["track_name", artist_col, "mood_score", "intensity_score"]]



