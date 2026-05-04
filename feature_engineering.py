from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        # Store threshold from training data
        self.area_threshold_ = X['area_sqft'].quantile(0.75)
        self.distance_median_ = X['distance_to_city_km'].median()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Feature engineering
        X['distance_to_city_km'] = X['distance_to_city_km'].replace(0, self.distance_median_)
        X['house_age'] = datetime.now().year - X['year_built']
        X['area_per_room'] = X['area_sqft'] / (X['bedrooms'] + X['bathrooms'])
        
        # Use stored threshold (IMPORTANT)
        X['is_luxury'] = (
            (X['area_sqft'] > self.area_threshold_) &
            (X['bathrooms'] >= 3) &
            (X['garage'] == 1)
        ).astype(int)
        
        return X