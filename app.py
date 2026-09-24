
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    movies = pd.read_csv("app_data/movies_app.csv")

    movies["genres_text"] = (
    movies["genres"].str.replace("|", " ", regex=False)
)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies["genres_text"])

    similarity = cosine_similarity(tfidf_matrix)

    return movies, similarity


movies, content_similarity = load_data()

movie_indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()


def recommend_movies(title, num_recommendations=10):

    idx = movie_indices[title]

    similarity_scores = list(
        enumerate(content_similarity[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[
        1:num_recommendations + 1
    ]

    movie_indices_list = [
        i[0] for i in similarity_scores
    ]

    return movies.iloc[movie_indices_list][
        ["title", "genres"]
    ]


st.title("🎬 Movie Recommendation System")

st.markdown(
    """
    ### 🍿 Discover your next favorite movie
    Find movies similar to the ones you love using
    **content-based recommendation** powered by TF-IDF and cosine similarity.
    """
)

st.divider()

st.subheader("🎥 Choose a movie")

selected_movie = st.selectbox(
    "Select a movie you like",
    movies["title"].sort_values().tolist()
)

num_recommendations = st.slider(
    "How many recommendations?",
    min_value=5,
    max_value=20,
    value=10
)

if st.button(
    "✨ Recommend Movies",
    type="primary",
    use_container_width=True
):

    recommendations = recommend_movies(
        selected_movie,
        num_recommendations
    )

    st.subheader("🍿 Recommended Movies")

    for i, (_, row) in enumerate(recommendations.iterrows(), start=1):

        with st.container(border=True):

            st.markdown(
                f"### {i}. 🍿 {row['title']}"
            )

            st.caption(
                f"🎭 Genres: {row['genres']}"
            )

st.divider()

with st.expander("ℹ️ About this project"):
    st.write(
        "This movie recommendation system uses content-based filtering "
        "with TF-IDF and cosine similarity. The project is built with "
        "Python, Pandas, Scikit-learn, and Streamlit."
    )