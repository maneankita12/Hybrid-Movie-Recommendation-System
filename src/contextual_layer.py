import pandas as pd
import numpy as np
from datetime import datetime


class ContextualLayer:
    def __init__(self):
        self.movies_df = None
        self.genre_time_mapping = {
            'morning': ['Documentary', 'Animation', 'Children', 'Family'],
            'afternoon': ['Comedy', 'Adventure', 'Musical', 'Fantasy'],
            'evening': ['Drama', 'Romance', 'Mystery', 'Thriller'],
            'night': ['Horror', 'Sci-Fi', 'Action', 'Crime']
        }
        
        self.genre_weekend_mapping = {
            'weekend': ['Action', 'Adventure', 'Sci-Fi', 'Fantasy', 'Animation'],
            'weekday': ['Drama', 'Documentary', 'Romance', 'Comedy']
        }
    
    def load_data(self, movies_path, ratings_path):
        self.movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'rating_count', 'avg_rating']
        
        self.movies_df = self.movies_df.merge(movie_stats, on='movieId', how='left')
        self.movies_df['rating_count'] = self.movies_df['rating_count'].fillna(0)
        self.movies_df['avg_rating'] = self.movies_df['avg_rating'].fillna(0)
        
        max_count = self.movies_df['rating_count'].max()
        self.movies_df['popularity_score'] = (
            self.movies_df['rating_count'] / max_count
        ) * self.movies_df['avg_rating'] / 5.0
        
        return self.movies_df
    
    def get_time_of_day(self):
        hour = datetime.now().hour
        
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def is_weekend(self):
        return datetime.now().weekday() >= 5
    
    def get_genre_boost(self, genres_str, time_context='auto', weekend_context='auto'):
        if not isinstance(genres_str, str):
            return 0.0
        
        if time_context == 'auto':
            time_context = self.get_time_of_day()
        
        if weekend_context == 'auto':
            weekend_context = 'weekend' if self.is_weekend() else 'weekday'
        elif weekend_context is True:
            weekend_context = 'weekend'
        else:
            weekend_context = 'weekday'
        
        genres = genres_str.split('|')
        
        time_boost = 0.0
        preferred_genres = self.genre_time_mapping.get(time_context, [])
        for genre in genres:
            if genre in preferred_genres:
                time_boost += 0.1
        
        weekend_boost = 0.0
        preferred_genres = self.genre_weekend_mapping.get(weekend_context, [])
        for genre in genres:
            if genre in preferred_genres:
                weekend_boost += 0.05
        
        return min(time_boost + weekend_boost, 0.3)  # Cap at 30% boost
    
    def apply_contextual_boost(self, recommendations, time_context='auto', 
                               weekend_context='auto', use_popularity=True):
        boosted_recs = []
        
        for rec in recommendations:
            movie_id = rec['movieId']
            base_score = rec.get('similarity_score', rec.get('predicted_rating', 0))
            
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            
            if movie_info.empty:
                boosted_recs.append(rec)
                continue
            
            movie_info = movie_info.iloc[0]
            
            genre_boost = self.get_genre_boost(
                movie_info['genres'], 
                time_context, 
                weekend_context
            )
            
            popularity_boost = 0.0
            if use_popularity:
                popularity_boost = movie_info['popularity_score'] * 0.15  # Max 15%
            
            final_score = base_score * (1 + genre_boost + popularity_boost)
            
            boosted_rec = rec.copy()
            boosted_rec['base_score'] = base_score
            boosted_rec['genre_boost'] = genre_boost
            boosted_rec['popularity_boost'] = popularity_boost
            boosted_rec['final_score'] = final_score
            boosted_rec['popularity'] = int(movie_info['rating_count'])
            boosted_rec['avg_rating'] = float(movie_info['avg_rating'])
            
            boosted_recs.append(boosted_rec)
        
        boosted_recs.sort(key=lambda x: x['final_score'], reverse=True)
        
        return boosted_recs


if __name__ == "__main__":
    ctx = ContextualLayer()
    ctx.load_data('data/movies.csv', 'data/ratings.csv')
    
    sample_recs = [
        {'movieId': 1, 'title': 'Toy Story (1995)', 'similarity_score': 0.8},
        {'movieId': 2, 'title': 'Jumanji (1995)', 'similarity_score': 0.75},
        {'movieId': 356, 'title': 'Forrest Gump (1994)', 'similarity_score': 0.7},
        {'movieId': 296, 'title': 'Pulp Fiction (1994)', 'similarity_score': 0.65},
        {'movieId': 318, 'title': 'The Shawshank Redemption (1994)', 'similarity_score': 0.6}
    ]
    
    for i, rec in enumerate(sample_recs, 1):
        print(f"{i}. {rec['title']:<40} | Score: {rec['similarity_score']:.4f}")
    
    boosted = ctx.apply_contextual_boost(sample_recs)
    
    for i, rec in enumerate(boosted, 1):
        print(f"{i}. {rec['title']:<40} | "
              f"Base: {rec['base_score']:.3f} | "
              f"Genre+: {rec['genre_boost']:.2f} | "
              f"Pop+: {rec['popularity_boost']:.2f} | "
              f"Final: {rec['final_score']:.3f}")
