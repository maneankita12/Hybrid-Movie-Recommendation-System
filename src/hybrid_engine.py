import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from collaborative_filtering import CollaborativeFilteringEngine
from content_based import ContentBasedEngine
from contextual_layer import ContextualLayer


class HybridRecommendationEngine:
    def __init__(self):
        self.cf_engine = CollaborativeFilteringEngine()
        self.cb_engine = ContentBasedEngine()
        self.ctx_layer = ContextualLayer()
        
        self.cf_loaded = False
        self.cb_loaded = False
        self.ctx_loaded = False
        
    def load_models(self, cf_path, cb_path):
        try:
            self.cf_engine.load_model(cf_path)
            self.cf_loaded = True
        except Exception as e:
            print(f"CF model not loaded: {e}")
        
        try:
            self.cb_engine.load_model(cb_path)
            self.cb_loaded = True
        except Exception as e:
            print(f" CB model not loaded: {e}")
        
        try:
            self.ctx_layer.load_data('data/movies.csv', 'data/ratings.csv')
            self.ctx_loaded = True
        except Exception as e:
            print(f"Context layer not loaded: {e}")
        
        print(f"✅ Models loaded: CF={self.cf_loaded}, CB={self.cb_loaded}, CTX={self.ctx_loaded}")
    
    def _normalize_scores(self, recommendations, score_key='similarity_score'):
        if not recommendations:
            return recommendations
        
        scores = [r.get(score_key, 0) for r in recommendations]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return recommendations
        
        for rec in recommendations:
            original = rec.get(score_key, 0)
            rec[f'{score_key}_normalized'] = (original - min_score) / (max_score - min_score)
        
        return recommendations
    
    def _merge_recommendations(self, cf_recs, cb_recs, cf_weight=0.6, cb_weight=0.4):
        cf_recs = self._normalize_scores(cf_recs, 'similarity_score')
        cb_recs = self._normalize_scores(cb_recs, 'similarity_score')
        
        cf_dict = {
            r['movieId']: r.get('similarity_score_normalized', r.get('similarity_score', 0))
            for r in cf_recs
        }
        
        cb_dict = {
            r['movieId']: r.get('similarity_score_normalized', r.get('similarity_score', 0))
            for r in cb_recs
        }

        all_movie_ids = set(cf_dict.keys()) | set(cb_dict.keys())
        
        merged = []
        for movie_id in all_movie_ids:
            cf_score = cf_dict.get(movie_id, 0)
            cb_score = cb_dict.get(movie_id, 0)
 
            hybrid_score = cf_weight * cf_score + cb_weight * cb_score
            
            movie_info = None
            for r in cf_recs + cb_recs:
                if r['movieId'] == movie_id:
                    movie_info = r
                    break
            
            if movie_info:
                merged.append({
                    'movieId': movie_id,
                    'title': movie_info['title'],
                    'similarity_score': hybrid_score,
                    'cf_score': cf_score,
                    'cb_score': cb_score,
                    'method': 'hybrid'
                })
        
        merged.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return merged
    
    def get_smart_recommendations(self, movie_id=None, user_id=None, top_k=10,
                                  use_context=True, strategy='adaptive'):
        if not movie_id and not user_id:
            return {'error': 'Must provide either movie_id or user_id'}
        if strategy == 'adaptive':
            cf_weight = 0.6 if self.cf_loaded else 0.0
            cb_weight = 0.4 if self.cb_loaded else 0.0
        
            if cf_weight == 0 and cb_weight > 0:
                cb_weight = 1.0
            elif cb_weight == 0 and cf_weight > 0:
                cf_weight = 1.0
                
        elif strategy == 'cf_heavy':
            cf_weight, cb_weight = 0.7, 0.3
        elif strategy == 'cb_heavy':
            cf_weight, cb_weight = 0.3, 0.7
        else: 
            cf_weight, cb_weight = 0.5, 0.5
        
        cf_recs = []
        cb_recs = []
        
        if movie_id:
            if self.cf_loaded:
                cf_recs = self.cf_engine.get_similar_movies(movie_id, top_k=top_k*2)
            
            if self.cb_loaded:
                cb_recs = self.cb_engine.get_similar_movies(movie_id, top_k=top_k*2)
        
        elif user_id:
            if self.cf_loaded:
                cf_recs = self.cf_engine.recommend_for_user(user_id, top_k=top_k*2)

            cf_weight = 1.0
            cb_weight = 0.0
        
        if not cf_recs and cb_recs:
            recommendations = cb_recs[:top_k]
            used_strategy = 'content_only'
        elif cf_recs and not cb_recs:
            recommendations = cf_recs[:top_k]
            used_strategy = 'collaborative_only'
        elif cf_recs and cb_recs:
            recommendations = self._merge_recommendations(cf_recs, cb_recs, cf_weight, cb_weight)
            recommendations = recommendations[:top_k*2] 
            used_strategy = f'hybrid_cf{int(cf_weight*100)}_cb{int(cb_weight*100)}'
        else:
            return {'error': 'No recommendations available'}
        
        if use_context and self.ctx_loaded:
            recommendations = self.ctx_layer.apply_contextual_boost(recommendations)
            used_strategy += '_context'
        
        recommendations = recommendations[:top_k]
        
        return {
            'recommendations': recommendations,
            'strategy': used_strategy,
            'weights': {'cf': cf_weight, 'cb': cb_weight},
            'count': len(recommendations)
        }
    
    def compare_methods(self, movie_id, top_k=5):
        movie_title = self.cf_engine.movie_id_to_title.get(movie_id, 'Unknown')
        print(f"Target Movie: {movie_title} (ID: {movie_id})\n")
        
        results = {}
    
        if self.cf_loaded:
            cf_recs = self.cf_engine.get_similar_movies(movie_id, top_k=top_k)
            results['Collaborative Filtering'] = cf_recs
        
        if self.cb_loaded:
            cb_recs = self.cb_engine.get_similar_movies(movie_id, top_k=top_k)
            results['Content-Based'] = cb_recs
        
        hybrid_result = self.get_smart_recommendations(movie_id=movie_id, top_k=top_k, use_context=False)
        if 'recommendations' in hybrid_result:
            results['Hybrid (No Context)'] = hybrid_result['recommendations']
        
        hybrid_ctx_result = self.get_smart_recommendations(movie_id=movie_id, top_k=top_k, use_context=True)
        if 'recommendations' in hybrid_ctx_result:
            results['Hybrid + Context'] = hybrid_ctx_result['recommendations']
        
        for method, recs in results.items():
            print(f"\n{method}:")
            print("-" * 100)
            for i, rec in enumerate(recs, 1):
                score = rec.get('final_score', rec.get('similarity_score', rec.get('predicted_rating', 0)))
                print(f"{i:2d}. {rec['title']:<50} | Score: {score:.4f}")
        
        return results


if __name__ == "__main__":
    hybrid = HybridRecommendationEngine()
    
    hybrid.load_models(
        'models/collaborative_filtering_model.pkl',
        'models/content_based_model.pkl'
    )
    
    toy_story_id = 1
    results = hybrid.compare_methods(toy_story_id, top_k=10)

    smart_result = hybrid.get_smart_recommendations(
        movie_id=toy_story_id,
        top_k=10,
        use_context=True,
        strategy='adaptive'
    )
    
    for i, rec in enumerate(smart_result['recommendations'], 1):
        score = rec.get('final_score', rec.get('similarity_score', 0))
        print(f"{i:2d}. {rec['title']:<50} | Score: {score:.4f}")
    
