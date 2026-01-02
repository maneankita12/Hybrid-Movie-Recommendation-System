import pandas as pd
import numpy as np
import requests
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os



class ContentBasedEngine:
    
    def __init__(self, tmdb_api_key=None):
        self.tmdb_api_key = tmdb_api_key or os.getenv('TMDB_API_KEY')
        self.movies_df = None
        self.enriched_movies = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.movie_indices = {}
        
    def load_data(self, movies_path, links_path):
        self.movies_df = pd.read_csv(movies_path)
        links_df = pd.read_csv(links_path)
        
        self.movies_df = self.movies_df.merge(links_df, on='movieId', how='left')
        
        print(f"Loaded {len(self.movies_df)} movies")
        print(f"Movies with TMDB ID: {self.movies_df['tmdbId'].notna().sum()}")
        
        return self.movies_df
    
    def fetch_tmdb_details(self, tmdb_id):
        if pd.isna(tmdb_id):
            return None
        
        try:
            url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
            params = {
                'api_key': self.tmdb_api_key,
                'append_to_response': 'credits,keywords'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
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
                    'genres': genres,
                    'cast': cast,
                    'director': director,
                    'keywords': keywords,
                    'overview': data.get('overview', ''),
                    'vote_average': data.get('vote_average', 0),
                    'popularity': data.get('popularity', 0)
                }
            
            return None
            
        except Exception as e:
            return None
    
    def enrich_movies(self, sample_size=None, cache_path='data/enriched_movies.pkl'):
        if os.path.exists(cache_path):
            self.enriched_movies = pd.read_pickle(cache_path)
            return self.enriched_movies

        movies_to_enrich = self.movies_df[self.movies_df['tmdbId'].notna()].copy()
        
        if sample_size:
            movies_to_enrich = movies_to_enrich.sample(min(sample_size, len(movies_to_enrich)))
        
        enriched_data = []
        total = len(movies_to_enrich)
        
        for idx, (_, row) in enumerate(movies_to_enrich.iterrows(), 1):
            if idx % 50 == 0:
                print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
            
            tmdb_data = self.fetch_tmdb_details(row['tmdbId'])
            
            enriched_row = {
                'movieId': row['movieId'],
                'title': row['title'],
                'genres': row['genres'],
                'tmdb_genres': tmdb_data['genres'] if tmdb_data else [],
                'cast': tmdb_data['cast'] if tmdb_data else [],
                'director': tmdb_data['director'] if tmdb_data else None,
                'keywords': tmdb_data['keywords'] if tmdb_data else [],
                'overview': tmdb_data['overview'] if tmdb_data else '',
                'vote_average': tmdb_data['vote_average'] if tmdb_data else 0,
                'popularity': tmdb_data['popularity'] if tmdb_data else 0
            }
            
            enriched_data.append(enriched_row)
        
        self.enriched_movies = pd.DataFrame(enriched_data)
        
        self.enriched_movies.to_pickle(cache_path)
        
        return self.enriched_movies
    
    def create_content_features(self):
        if self.enriched_movies is None:
            raise ValueError("Must enrich movies first!")
        
        def create_feature_string(row):
            parts = []
            
            if isinstance(row['genres'], str):
                parts.extend(row['genres'].split('|') * 3)
            parts.extend(row['tmdb_genres'] * 3)

            if row['cast']:
                parts.extend(row['cast'][:3] * 2) 
                parts.extend(row['cast'][3:])
            
            if row['director']:
                parts.extend([row['director']] * 2)
            
            parts.extend(row['keywords'] * 2)
            
            if row['overview']:
                parts.append(row['overview'])
            
            return ' '.join(str(p).lower() for p in parts if p)
        
        self.enriched_movies['content_features'] = self.enriched_movies.apply(
            create_feature_string, axis=1
        )
        
        self.movie_indices = {
            row['movieId']: idx 
            for idx, row in self.enriched_movies.iterrows()
        }
        
        return self.enriched_movies
    
    def build_similarity_matrix(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.enriched_movies['content_features']
        )
        
    def get_similar_movies(self, movie_id, top_k=10):
        if movie_id not in self.movie_indices:
            return []
        
        idx = self.movie_indices[movie_id]
        
        movie_vector = self.tfidf_matrix[idx]
        similarities = cosine_similarity(movie_vector, self.tfidf_matrix)[0]
        
        similar_indices = similarities.argsort()[::-1][1:top_k+1]
        
        results = []
        for sim_idx in similar_indices:
            movie_data = self.enriched_movies.iloc[sim_idx]
            results.append({
                'movieId': movie_data['movieId'],
                'title': movie_data['title'],
                'similarity_score': float(similarities[sim_idx]),
                'genres': movie_data['genres'],
                'method': 'content_based'
            })
        
        return results
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'enriched_movies': self.enriched_movies,
                'tfidf_matrix': self.tfidf_matrix,
                'vectorizer': self.vectorizer,
                'movie_indices': self.movie_indices
            }, f)
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.enriched_movies = data['enriched_movies']
            self.tfidf_matrix = data['tfidf_matrix']
            self.vectorizer = data['vectorizer']
            self.movie_indices = data['movie_indices']

if __name__ == "__main__":
    cb_engine = ContentBasedEngine()
    
    cb_engine.load_data('data/movies.csv', 'data/links.csv')
    
    cb_engine.enrich_movies(sample_size=500) 
    
    cb_engine.create_content_features()
    
    cb_engine.build_similarity_matrix()
    
    toy_story_id = 1
    similar = cb_engine.get_similar_movies(toy_story_id, top_k=10)
    
    for i, movie in enumerate(similar, 1):
        print(f"{i:2d}. {movie['title']:<50} | Score: {movie['similarity_score']:.4f}")
    
    cb_engine.save_model('models/content_based_model.pkl')
