from pathlib import Path
import pandas as pd
import streamlit as st
from recommendation import RecommendationEngine

st.set_page_config(page_title="FRAME · Find your next film", page_icon="🎬", layout="wide")
DATA_DIR = Path(__file__).resolve().parent / "app_data"


@st.cache_resource
def load_engine(artifact_stamp):
    try:
        return RecommendationEngine.from_directory(DATA_DIR), None
    except (ValueError, OSError, KeyError):
        return RecommendationEngine(pd.read_csv(DATA_DIR / "movies_app.csv")), (
            "Viewer-based matches are temporarily unavailable. Showing genre matches."
        )


artifact_stamp = tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size)
                       for path in sorted(DATA_DIR.iterdir()) if path.is_file())
engine, notice = load_engine(artifact_stamp)
movies = engine.movies
if notice:
    st.warning(notice)

from html import escape
import re

st.markdown(
    """
    <style>
    .block-container { max-width: 1160px; padding-top: 3rem; padding-bottom: 3rem; }
    .film-brand { display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 1.2rem; border-bottom: 1px solid #e3ddd4; gap: 1rem; }
    .film-wordmark { font-size: 1.1rem; font-weight: 750; letter-spacing: .16em; }
    .film-note, .film-eyebrow { color: #716b65; font-size: .8rem; }
    .film-eyebrow { text-transform: uppercase; letter-spacing: .12em; margin-bottom: .8rem; }
    .film-hero { padding: 3.2rem 0 1.8rem; }
    .film-hero h1 { font-family: Georgia, serif; font-weight: 400; font-size: clamp(2.6rem, 6vw, 4.6rem);
        line-height: 1.08; letter-spacing: -.045em; margin: 0 0 1rem; max-width: 740px; }
    .film-hero p { color: #716b65; font-size: 1.08rem; line-height: 1.65; max-width: 540px; }
    .film-section { margin: 2.8rem 0 1.4rem; }
    .film-section h2 { font-family: Georgia, serif; font-size: 1.9rem; font-weight: 400; margin: .35rem 0; }
    .film-section p { color: #716b65; margin: .5rem 0; }
    .film-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
    .film-card { background: #fff; border: 1px solid #e3ddd4; border-radius: 8px;
        padding: 1.4rem; min-height: 225px; display: flex; flex-direction: column; }
    .film-card-top { display: flex; justify-content: space-between; align-items: center;
        color: #716b65; font-size: .78rem; margin-bottom: 1.3rem; }
    .film-rank { color: #8b343b; font-weight: 650; letter-spacing: .08em; }
    .film-card h3 { font-family: Georgia, serif; font-size: 1.35rem; font-weight: 400;
        line-height: 1.35; margin: 0 0 .85rem; overflow-wrap: anywhere; }
    .film-genres { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: 1rem; }
    .film-genre { font-size: .73rem; padding: .2rem .5rem; background: #f3efe9;
        border-radius: 4px; color: #534c46; }
    .film-reason { font-size: .8rem; line-height: 1.5; color: #716b65;
        margin-top: auto; padding-top: .7rem; border-top: 1px solid #eee9e1; }
    .film-empty { border-top: 1px solid #e3ddd4; margin-top: 2.5rem; padding: 2rem 0; }
    .film-empty h2 { font-family: Georgia, serif; font-weight: 400; font-size: 1.6rem; }
    .film-empty p { color: #716b65; max-width: 530px; line-height: 1.65; }
    .film-footer { border-top: 1px solid #e3ddd4; margin: 3rem 0 1rem;
        padding-top: 1rem; color: #716b65; font-size: .78rem; }
    @media (max-width: 850px) { .film-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 540px) {
        .block-container { padding-top: 1.5rem; }
        .film-grid { grid-template-columns: 1fr; }
        .film-hero { padding-top: 2rem; }
        .film-note { max-width: 130px; text-align: right; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<header class="film-brand"><span class="film-wordmark">FRAME</span>'
    '<span class="film-note">A little direction for movie night.</span></header>'
    '<section class="film-hero"><div class="film-eyebrow">Your next watch</div>'
    '<h1>Good films lead to<br>more good films.</h1>'
    '<p>Start with a film you love. Find something familiar, '
    'something unexpected, and something worth watching.</p></section>',
    unsafe_allow_html=True,
)

with st.form("film_search", border=False):
    selected_movie = st.selectbox(
        "Start with a film you like",
        movies["title"].sort_values().tolist(),
        help="Type a title to search the collection.",
    )
    num_recommendations = st.slider(
        "Number of films", min_value=5, max_value=20, value=10
    )
    discovery_mode = st.selectbox(
        "What would you like to explore?",
        ["Similar films", "Genre matches"],
        help="Similar films combines shared genres with MovieLens viewer-rating patterns.",
    )
    submitted = st.form_submit_button("Find films", type="primary")

if submitted:
    with st.spinner("Finding films with similar genres…"):
        try:
            st.session_state["film_results"] = engine.recommend(
                engine.find_movie_id(selected_movie),
                num_recommendations,
                mode="hybrid" if discovery_mode == "Similar films" else "genres",
            )
            st.session_state["film_source"] = selected_movie
            st.session_state["film_mode"] = discovery_mode
        except ValueError:
            st.error("That film could not be matched. Please select another title.")

if "film_results" in st.session_state:
    source = st.session_state["film_source"]
    recommendations = st.session_state["film_results"]
    based_on = (
        "shared genres and viewer-rating patterns"
        if st.session_state.get("film_mode") == "Similar films" and engine.neighbors
        else "shared genres"
    )
    st.markdown(
        '<section class="film-section"><div class="film-eyebrow">The next chapter</div>'
        f'<h2>Because you liked {escape(source)}</h2>'
        f'<p>{len(recommendations)} films to explore, based on {based_on}.</p></section>',
        unsafe_allow_html=True,
    )
    cards = []
    for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
        title = str(row["title"])
        title_parts = re.fullmatch(r"(.*)\s+\((\d{4})\)", title)
        display_title = title_parts.group(1) if title_parts else title
        year = title_parts.group(2) if title_parts else "Film"
        genres = str(row["genres"]).split("|")
        reason = str(row["reason"])
        rating_note = (
            f'{row["avg_rating"]:.1f}/5 · {int(row["rating_count"]):,} MovieLens ratings'
            if row["rating_count"] else "Discover this film"
        )
        tags = "".join(
            f'<span class="film-genre">{escape(genre)}</span>' for genre in genres
        )
        cards.append(
            '<article class="film-card">'
            f'<div class="film-card-top"><span class="film-rank">{rank:02d}</span>'
            f'<span>{escape(year)}</span></div>'
            f'<h3>{escape(display_title)}</h3>'
            f'<div class="film-genres">{tags}</div>'
            f'<div class="film-reason">{escape(reason)}<br>{escape(rating_note)}</div></article>'
        )
    st.markdown(
        '<div class="film-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<section class="film-empty"><div class="film-eyebrow">Where to begin</div>'
        '<h2>One favorite is all it takes.</h2>'
        '<p>Search for a film above, then choose how many recommendations you want. '
        'Your next watch might be just around the corner.</p></section>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<footer class="film-footer">FRAME · Built by Aahil · MovieLens collection</footer>',
    unsafe_allow_html=True,
)
with st.expander("How these recommendations work"):
    st.write(
        "Similar films combines shared genres with patterns in movies that "
        "MovieLens viewers liked, plus a small rating-quality signal. Genre matches "
        "uses shared genres with a rating-quality tie-breaker. Results are balanced to reduce repetition. "
        "The collection covers films released through 2000. This is film-to-film "
        "discovery, rather than a prediction of your personal rating."
    )
    if engine.features.has_details.any():
        st.caption("Available story, cast, director and keyword metadata also informs content comparisons.")
    else:
        st.caption(
            "Story and cast metadata is not available in this collection; matches use genres and viewer ratings."
        )

