import pandas as pd
import numpy as np
import pickle
# from surprise import Dataset, Reader, SVD
#from surprise.model_selection import train_test_split, cross_validate
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringEngine:
    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        
        self.model = None
        self.trainset = None
        self.movies_df = None
        
        self.movie_id_to_index = {}
        self.index_to_movie_id = {}
        self.movie_id_to_title = {}
        
    # def load_data(self, ratings_path, movies_path):
    #     ratings = pd.read_csv(ratings_path)
    #     self.movies_df = pd.read_csv(movies_path)
    
    #     self.movie_id_to_title = dict(zip(
    #         self.movies_df['movieId'], 
    #         self.movies_df['title']
    #     ))
        
    #     reader = Reader(rating_scale=(0.5, 5.0))
    #     data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        
    #     print(f"Loaded {len(ratings)} ratings for {len(self.movies_df)} movies")
    #     return data, ratings
    
    # def train(self, data):
    #     trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    #     self.trainset = trainset
        
    #     self.model = SVD(
    #         n_factors=self.n_factors,
    #         n_epochs=self.n_epochs,
    #         lr_all=self.lr_all,
    #         reg_all=self.reg_all,
    #         random_state=42,
    #         verbose=True
    #     )
        
    #     self.model.fit(trainset)
        
    #     for iid in trainset.all_items():
    #         raw_id = trainset.to_raw_iid(iid)
    #         self.movie_id_to_index[raw_id] = iid
    #         self.index_to_movie_id[iid] = raw_id
        
    #     predictions = self.model.test(testset)
        
    #     errors = [abs(pred.r_ui - pred.est) for pred in predictions]
    #     rmse = np.sqrt(np.mean([e**2 for e in errors]))
    #     mae = np.mean(errors)
        
    #     print(f"   RMSE: {rmse:.4f}")
    #     print(f"   MAE: {mae:.4f}")
        
    #     return {'rmse': rmse, 'mae': mae}
    
    def get_similar_movies(self, movie_id, top_k=10):
        if movie_id not in self.movie_id_to_index:
            return []
        
        movie_inner_id = self.movie_id_to_index[movie_id]
        movie_factors = self.model.qi[movie_inner_id].reshape(1, -1)
        
        all_factors = self.model.qi
        
        similarities = cosine_similarity(movie_factors, all_factors)[0]
        
        similar_indices = similarities.argsort()[::-1][1:top_k+1]
        
        results = []
        for idx in similar_indices:
            similar_movie_id = self.index_to_movie_id[idx]
            results.append({
                'movieId': similar_movie_id,
                'title': self.movie_id_to_title.get(similar_movie_id, 'Unknown'),
                'similarity_score': float(similarities[idx]),
                'method': 'collaborative_filtering'
            })
        
        return results
    
    def predict_rating(self, user_id, movie_id):
        if self.model is None:
            return None
        
        prediction = self.model.predict(user_id, movie_id)
        return {
            'user_id': user_id,
            'movie_id': movie_id,
            'predicted_rating': prediction.est,
            'title': self.movie_id_to_title.get(movie_id, 'Unknown')
        }
    
    def recommend_for_user(self, user_id, top_k=10, exclude_rated=True):
        if self.model is None:
            return []
        
        all_movie_ids = list(self.movie_id_to_title.keys())
        
        rated_movies = set()
        if exclude_rated:
            try:
                rated_movies = set(self.trainset.ur[self.trainset.to_inner_uid(user_id)])
                rated_movies = {self.index_to_movie_id[iid] for iid in rated_movies}
            except:
                pass

        predictions = []
        for movie_id in all_movie_ids:
            if movie_id not in rated_movies:
                pred = self.model.predict(user_id, movie_id)
                predictions.append({
                    'movieId': movie_id,
                    'title': self.movie_id_to_title[movie_id],
                    'predicted_rating': pred.est,
                    'method': 'collaborative_filtering'
                })
        
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return predictions[:top_k]
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'trainset': self.trainset,
                'movie_id_to_index': self.movie_id_to_index,
                'index_to_movie_id': self.index_to_movie_id,
                'movie_id_to_title': self.movie_id_to_title,
                'movies_df': self.movies_df
            }, f)
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.trainset = data['trainset']
            self.movie_id_to_index = data['movie_id_to_index']
            self.index_to_movie_id = data['index_to_movie_id']
            self.movie_id_to_title = data['movie_id_to_title']
            self.movies_df = data['movies_df']


if __name__ == "__main__":
    cf_engine = CollaborativeFilteringEngine(
        n_factors=50,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02
    )
    
    data, ratings = cf_engine.load_data(
        'data/ratings.csv',
        'data/movies.csv'
    )
    
    metrics = cf_engine.train(data)
    
    toy_story_id = 1
    similar = cf_engine.get_similar_movies(toy_story_id, top_k=10)
    
    for i, movie in enumerate(similar, 1):
        print(f"{i:2d}. {movie['title']:<50} | Score: {movie['similarity_score']:.4f}")
    
    user_recs = cf_engine.recommend_for_user(user_id=1, top_k=10)
    
    for i, movie in enumerate(user_recs, 1):
        print(f"{i:2d}. {movie['title']:<50} | Rating: {movie['predicted_rating']:.2f}")
    
    cf_engine.save_model('models/collaborative_filtering_model.pkl')
