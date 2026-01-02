import streamlit as st
import sys
sys.path.insert(0, 'src')

from hybrid_engine import HybridRecommendationEngine
from content_based import ContentBasedEngine
import pandas as pd
from fuzzywuzzy import process
import requests
import os

try:
    if 'TMDB_API_KEY' in st.secrets:
        os.environ['TMDB_API_KEY'] = st.secrets['TMDB_API_KEY']
except:
    pass

if os.path.exists('.env') and 'TMDB_API_KEY' not in os.environ:
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .stTable {
        width: 100%;
    }
    /* Increase sidebar width */
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 400px;
        max-width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"]{
        min-width: 400px;
        max-width: 400px;
        margin-left: -400px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engines():
    hybrid_engine = HybridRecommendationEngine()
    hybrid_engine.load_models(
        'models/collaborative_filtering_model.pkl',
        'models/content_based_model.pkl'
    )
    
    content_engine = ContentBasedEngine(tmdb_api_key=os.environ.get('TMDB_API_KEY'))
    content_engine.load_model('models/content_based_model.pkl')
    
    return hybrid_engine, content_engine

@st.cache_data
def load_movie_list():
    movies = pd.read_csv('data/movies.csv')
    return movies

def find_movie_in_dataset(movie_input, movies_df):
    all_titles = movies_df['title'].tolist()
    match = process.extractOne(movie_input, all_titles)
    
    if match and match[1] >= 70:  
        matched_title = match[0]
        movie_id = movies_df[movies_df['title'] == matched_title]['movieId'].values[0]
        return movie_id, matched_title, match[1], True
    return None, None, 0, False

def search_tmdb_movie(movie_name, api_key):
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': api_key,
            'query': movie_name
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                movie = data['results'][0]
                return {
                    'title': movie['title'],
                    'year': movie.get('release_date', '')[:4] if movie.get('release_date') else 'N/A',
                    'tmdb_id': movie['id'],
                    'overview': movie.get('overview', ''),
                    'genres': movie.get('genre_ids', [])
                }
        return None
    except Exception as e:
        st.error(f"TMDB Search Error: {str(e)}")
        return None

def get_tmdb_details(tmdb_id, api_key):
    """Get detailed movie info from TMDB"""
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {
            'api_key': api_key,
            'append_to_response': 'credits,keywords'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            genres = [g['name'] for g in data.get('genres', [])]
            
            cast = []
            if 'credits' in data and 'cast' in data['credits']:
                cast = [actor['name'] for actor in data['credits']['cast'][:5]]
            
            director = None
            if 'credits' in data and 'crew' in data['credits']:
                for person in data['credits']['crew']:
                    if person.get('job') == 'Director':
                        director = person['name']
                        break
            
            keywords = []
            if 'keywords' in data and 'keywords' in data['keywords']:
                keywords = [kw['name'] for kw in data['keywords']['keywords'][:10]]
            
            return {
                'title': data['title'],
                'genres': genres,
                'overview': data.get('overview', ''),
                'cast': cast,
                'director': director,
                'keywords': keywords,
                'vote_average': data.get('vote_average', 0),
                'popularity': data.get('popularity', 0)
            }
        return None
    except Exception as e:
        st.error(f"TMDB Details Error: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">AI Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    with st.expander("About this System", expanded=False):
        st.markdown("""
        #### Hybrid Recommendation System
        
        This system provides intelligent recommendations using three powerful techniques:
        
        **Collaborative Filtering** - Based on user behavior patterns and movie rating similarities
        
        **Content-Based Filtering** - Based on movie attributes & metadata (genres, cast, director, keywords)
        
        **Contextual Signals** - Adapts recommendations based on time of day and movie popularity
        
        ---
        Built with Python, Scikit-learn, Surprise, and Streamlit.
        """)
    
    with st.spinner("Loading recommendation engines..."):
        hybrid_engine, content_engine = load_engines()
        movies_df = load_movie_list()
    
    tmdb_api_key = os.environ.get('TMDB_API_KEY')
    
    # if tmdb_api_key:
    #     st.sidebar.success(f"✓ TMDB API configured (Key: ...{tmdb_api_key[-4:]})")
    # else:
    #     st.sidebar.warning("⚠ TMDB API not configured - limited to dataset movies only")
    
    # st.sidebar.header("Configuration")
    
    mode = st.sidebar.radio(
        "Recommendation Mode:",
        ["Similar Movies", "Personalized User Recommendations"],
        help="Get recommendations based on a movie or build a user profile"
    )
    
    movie_id = None
    tmdb_movie = None
    user_profile_movies = []
    selected_movie = None
    in_dataset = False
    
    if mode == "Similar Movies":
        st.sidebar.subheader("Movie Search")
        movie_input = st.sidebar.text_input(
            "Enter any movie title:",
            placeholder="e.g: Oppenheimer, Barbie, etc",
            help="Search any movie - from classics to new releases"
        )
        
        if movie_input:
            movie_id, matched_title, match_score, in_dataset = find_movie_in_dataset(movie_input, movies_df)
            
            if in_dataset:
                st.sidebar.success(f"✓ Found in database: {matched_title} (Match: {match_score}%)")
                selected_movie = matched_title
            else:
                if tmdb_api_key:
                    with st.spinner("Searching TMDB..."):
                        st.sidebar.info("Searching TMDB database...")
                        tmdb_result = search_tmdb_movie(movie_input, tmdb_api_key)
                    
                    if tmdb_result:
                        st.sidebar.success(f"✓ Found on TMDB: {tmdb_result['title']} ({tmdb_result['year']})")
                        st.sidebar.caption("This movie is not in our training dataset. We'll find similar movies based on content.")
                        tmdb_movie = get_tmdb_details(tmdb_result['tmdb_id'], tmdb_api_key)
                        selected_movie = f"{tmdb_result['title']} ({tmdb_result['year']})"
                    else:
                        st.sidebar.error(f"Movie '{movie_input}' not found on TMDB. Try a different title or check spelling.")
                else:
                    st.sidebar.error("TMDB API key not configured. Can only search movies in dataset.")
    
    else:
        # User profile builder
        st.sidebar.subheader("Build Your Profile")
        st.sidebar.markdown("Rate 5-10 movies you've watched to get personalized recommendations")
        
        num_movies = st.sidebar.number_input(
            "How many movies to rate?",
            min_value=5,
            max_value=15,
            value=5,
            help="Rating more movies gives better recommendations"
        )
        
        user_profile_movies = []
        
        for i in range(num_movies):
            col1, col2 = st.sidebar.columns([3, 1])
            
            with col1:
                movie_input = st.text_input(
                    f"Movie {i+1}:",
                    key=f"movie_{i}",
                    placeholder="Enter movie title"
                )
            
            with col2:
                rating = st.selectbox(
                    "Rating:",
                    options=[5, 4, 3, 2, 1],
                    index=0,
                    key=f"rating_{i}"
                )
            
            if movie_input:
                mid, mtitle, mscore, found = find_movie_in_dataset(movie_input, movies_df)
                if found:
                    user_profile_movies.append({
                        'movieId': mid,
                        'title': mtitle,
                        'rating': float(rating)
                    })
    
    with st.sidebar.expander("Advanced Settings"):
        top_k = st.slider("Number of recommendations:", 5, 20, 10)
    
    if st.sidebar.button("Get Recommendations", type="primary"):
        
        if mode == "Similar Movies" and not movie_id and not tmdb_movie:
            st.warning("Please enter a movie title to get recommendations.")
            return
        
        if mode == "Personalized User Recommendations" and len(user_profile_movies) < 3:
            st.warning("Please rate at least 3 movies to get personalized recommendations.")
            return
        
        with st.spinner("Analyzing and generating recommendations..."):
            
            if mode == "Similar Movies":
                if in_dataset:
                    result = hybrid_engine.get_smart_recommendations(
                        movie_id=movie_id,
                        user_id=None,
                        top_k=top_k,
                        use_context=True,
                        strategy='adaptive'
                    )
                    result['source'] = 'hybrid'
                else:
                    if tmdb_movie:
                        similar_movies = content_engine.find_similar_in_dataset(tmdb_movie, top_k=top_k)
                        
                        similar_movies = hybrid_engine.ctx_layer.apply_contextual_boost(similar_movies)
                        
                        result = {
                            'recommendations': similar_movies,
                            'strategy': 'content_based_tmdb',
                            'weights': {'cf': 0, 'cb': 1.0},
                            'count': len(similar_movies),
                            'source': 'tmdb'
                        }
                    else:
                        st.error("Could not fetch movie details from TMDB.")
                        return
                
            else:
                st.info(f"Analyzing your profile based on {len(user_profile_movies)} rated movies...")
                
                highest_rated = max(user_profile_movies, key=lambda x: x['rating'])
                
                result = hybrid_engine.get_smart_recommendations(
                    movie_id=highest_rated['movieId'],
                    user_id=None,
                    top_k=top_k,
                    use_context=True,
                    strategy='cb_heavy'
                )
                
                rated_ids = {m['movieId'] for m in user_profile_movies}
                result['recommendations'] = [
                    r for r in result['recommendations'] 
                    if r['movieId'] not in rated_ids
                ]
                result['source'] = 'profile'
        
        if 'error' in result:
            st.error(f"Error: {result['error']}")
            return
        
        if mode == "Similar Movies":
            st.subheader(f"Movies similar to: {selected_movie}")
            
            if result.get('source') == 'tmdb' and tmdb_movie:
                with st.expander("Movie Details", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**Genres:** {', '.join(tmdb_movie['genres'])}")
                        if tmdb_movie.get('director'):
                            st.markdown(f"**Director:** {tmdb_movie['director']}")
                        if tmdb_movie.get('cast'):
                            st.markdown(f"**Cast:** {', '.join(tmdb_movie['cast'][:3])}")
                    with col2:
                        st.metric("TMDB Rating", f"{tmdb_movie['vote_average']:.1f}/10")
                    
                    if tmdb_movie.get('overview'):
                        st.markdown(f"**Overview:** {tmdb_movie['overview']}")
        else:
            st.subheader("Personalized Recommendations Based on Your Profile")
            
            with st.expander("Your Rated Movies"):
                profile_df = pd.DataFrame(user_profile_movies)
                profile_df = profile_df.sort_values('rating', ascending=False)
                st.dataframe(profile_df[['title', 'rating']], use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Strategy: {result['strategy'].replace('_', ' ').title()}")
        with col2:
            st.caption(f"Weights: CF {result['weights']['cf']:.0%} / CB {result['weights']['cb']:.0%}")
        
        st.markdown("---")
        
        recommendations = result['recommendations']
        
        if not recommendations:
            st.warning("No recommendations found. Try adjusting your search.")
            return
        
        table_data = []
        for i, rec in enumerate(recommendations, 1):
            score = rec.get('final_score', rec.get('similarity_score', 0))
            
            table_data.append({
                'Rank': i,
                'Title': rec['title'],
                'Score': f"{score:.4f}",
                'Popularity': rec.get('popularity', 'N/A'),
                'Avg Rating': f"{rec.get('avg_rating', 0):.2f}" if rec.get('avg_rating') else 'N/A'
            })
        
        df = pd.DataFrame(table_data)
        
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Title": st.column_config.TextColumn("Title", width="large"),
                "Score": st.column_config.TextColumn("Score", width="small"),
                "Popularity": st.column_config.NumberColumn("Popularity", width="small"),
                "Avg Rating": st.column_config.TextColumn("Rating", width="small")
            }
        )


if __name__ == "__main__":
    main()
