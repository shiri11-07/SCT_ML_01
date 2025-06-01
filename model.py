import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load data
df = pd.read_csv('house_prices.csv')

# Features and target
X = df[['sqft', 'bedrooms', 'bathrooms']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as house_price_model.pkl")
