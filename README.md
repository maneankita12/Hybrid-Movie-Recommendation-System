# AI Movie Recommendation System

A hybrid movie recommendation system combining Collaborative Filtering, Content-Based Filtering, and Contextual Signals to provide personalized movie recommendations.

## Features

- **Hybrid Recommendation Engine**: Combines multiple AI techniques for accurate recommendations
- **Search Any Movie**: Search from 9,700+ movies in dataset or any movie via TMDB API
- **Personalized Profiles**: Rate movies to build your taste profile
- **Contextual Awareness**: Adapts recommendations based on time of day and popularity
- **Clean UI**: Simple, professional Streamlit interface

## System Architecture

### 1. Collaborative Filtering (SVD)
- Matrix Factorization using Singular Value Decomposition
- Learns latent factors from user-movie interactions
- RMSE: 0.8775, MAE: 0.6742

### 2. Content-Based Filtering
- Uses movie metadata: genres, cast, director, keywords, plot
- TF-IDF vectorization with cosine similarity
- Enriched with TMDB API data for 9,734 movies

### 3. Contextual Layer
- Time-of-day genre boosting
- Popularity weighting
- Adaptive weight adjustment

### 4. Hybrid Fusion
- Adaptive weighting (60% CF + 40% Content by default)
- Intelligent fallback for cold-start scenarios

## Installation

### Prerequisites
- Python 3.9+
- TMDB API Key ([Get it here](https://www.themoviedb.org/settings/api))

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/MovieRecommenderFromScratch.git
cd MovieRecommenderFromScratch
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your TMDB_API_KEY
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage

### Similar Movies Mode
1. Enter any movie title in the search box
2. System will find it in dataset or search TMDB
3. Click "Get Recommendations"
4. View personalized similar movies

### Personalized User Mode
1. Select "Personalized User Recommendations"
2. Rate 5-10 movies you've watched (1-5 stars)
3. Click "Get Recommendations"
4. Get recommendations based on your taste profile

## Project Structure
```
MovieRecommenderFromScratch/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Dependencies
├── .env.example                       # Environment template
│
├── data/                              # Dataset
│   ├── movies.csv                     # 9,742 movies
│   ├── ratings.csv                    # 100K+ ratings
│   ├── links.csv                      # TMDB/IMDB IDs
│   └── enriched_movies.pkl           # TMDB metadata cache
│
├── models/                            # Trained models
│   ├── collaborative_filtering_model.pkl
│   └── content_based_model.pkl
│
└── src/                               # Source modules
    ├── collaborative_filtering.py     # SVD implementation
    ├── content_based.py              # Content-based filtering
    ├── contextual_layer.py           # Contextual boosting
    └── hybrid_engine.py              # Hybrid fusion engine
```

## Technologies Used

- **Python 3.9+**
- **Streamlit** - Web interface
- **Scikit-learn** - ML algorithms, TF-IDF, cosine similarity
- **Scikit-surprise** - Collaborative filtering (SVD)
- **Pandas & NumPy** - Data processing
- **TMDB API** - Movie metadata enrichment
- **FuzzyWuzzy** - Fuzzy string matching

## Dataset

- **MovieLens Small Dataset** (100K ratings)
- 9,742 movies, 610 users
- Enriched with TMDB metadata (cast, director, keywords, genres)
