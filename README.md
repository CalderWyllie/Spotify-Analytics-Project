🎧 Spotify Song Popularity Prediction & Recommendation System

This project analyzes a dataset of popular Spotify songs and builds two main components:

A machine learning model that predicts how many streams a song will receive

A content-based recommendation system that finds songs with similar audio characteristics

The goal was to explore what drives song popularity and how audio features can be used to recommend similar “vibe” tracks.

📂 Project Overview
1. Data Cleaning

Removed commas from numeric columns (e.g., “1,021” → 1021)

Converted percentage columns into decimals

Filled missing values with median values

Standardized column names

Engineered new features:

mood_score = valence × danceability

intensity_score = combination of energy, danceability, and BPM

2. Popularity Prediction (Random Forest)

I trained a Random Forest regression model using audio features, metadata, and playlist/chart metrics.

Model Performance

R² Score: 0.835

MAE: ~140 million streams

Top Predictors

in_deezer_playlists
in_spotify_playlists
in_apple_playlists
released_year
in_spotify_charts

Playlist exposure ended up being the strongest signal for popularity.

🎵 3. Content-Based Song Recommender

The recommendation system uses cosine similarity on standardized audio features:

danceability
energy
valence
acousticness
instrumentalness
liveness
speechiness
bpm
mood_score
intensity_score

Example:

recommend("Cruel Summer", n=4)


Sample Output

Top 4 recommendations similar to 'Cruel Summer':

                   track_name     artists_name  mood_score  intensity_score
                    die first    Nessa Barrett      0.1936         0.622835
                     good 4 u   Olivia Rodrigo      0.3808         0.644583
                          O.O            NMIXX      0.1092         0.676087
                   Prohibidox             Feid      0.3380         0.762379
                 


This recommends songs with a similar “vibe,” not necessarily the same genre.
