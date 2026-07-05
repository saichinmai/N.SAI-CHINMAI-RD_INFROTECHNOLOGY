import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page settings
st.set_page_config(
    page_title="AI Recommendation System",
    layout="centered"
)

# Title
st.title("AI-Based Movie Recommendation System")
st.write("Get smart movie recommendations")

# Load dataset
movies = pd.read_csv("movies.csv")

# Create combined features
movies["content"] = (
    movies["Genre"] + " " +
    movies["Description"]
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    movies["content"]
)

# Similarity matrix
similarity = cosine_similarity(
    tfidf_matrix
)

# Dropdown
selected_movie = st.selectbox(
    "Select a Movie",
    movies["Movie"]
)

# Recommendation function
def recommend(movie):

    movie_index = movies[
        movies["Movie"] == movie
    ].index[0]

    similarity_scores = list(
        enumerate(
            similarity[movie_index]
        )
    )

    sorted_movies = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommended_movies = []

    for i in sorted_movies:
        recommended_movies.append(
            movies.iloc[i[0]].Movie
        )

    return recommended_movies


# Button
if st.button("Recommend"):

    recommendations = recommend(
        selected_movie
    )

    st.subheader(
        "Recommended Movies"
    )

    for movie in recommendations:
        st.success(movie)