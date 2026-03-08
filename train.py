import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

read_df = pd.read_csv('House_Rent_Dataset.csv')

features = ['BHK', 'Size', 'Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Bathroom']
target = 'Rent'

X = read_df[features]
y = read_df[target]

categorical_features = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred']
numerical_features = ['BHK', 'Size', 'Bathroom']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

with open('house_rent_model.pkl', 'wb') as f:
    pickle.dump(model, f)
