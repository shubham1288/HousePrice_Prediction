import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from feature_engineering import FeatureEngineer


# Load data
df = pd.read_csv("HousePrice_POC.csv")

# Clean data
df = df.drop_duplicates()
df['school_rating'].fillna(0, inplace=True)
df['distance_to_city_km'].fillna(df['distance_to_city_km'].median(), inplace=True)

df = df.drop('house_id', axis=1)

# Split
X = df.drop('price_inr', axis=1)
y = df['price_inr']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Columns
num_cols = [
    'area_sqft','bedrooms','bathrooms','floors',
    'year_built','garage','distance_to_city_km',
    'school_rating','renovated',
    'house_age','area_per_room','is_luxury'
]

cat_cols = ['locality','condition']

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

# Pipeline
random_forest_pipeline = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(n_estimators=300, min_samples_split=10,min_samples_leaf=4,random_state=42))
])

# Train Random Forest
random_forest_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = random_forest_pipeline.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

# Save model
joblib.dump(random_forest_pipeline, "house_price_model_rf.pkl")

print("Model saved successfully!")




