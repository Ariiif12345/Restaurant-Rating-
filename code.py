import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# KFC API data extraction (Example URL, replace with actual KFC API URL)
kfc_api_url = "https://api.kfc.com/example_endpoint"
response = requests.get(kfc_api_url)
kfc_data = response.json()

# Extract relevant fields
restaurants = []
for restaurant in kfc_data['restaurants']:
    restaurants.append({
        'name': restaurant['name'],
        'location': restaurant['location'],
        'rating': restaurant['rating'],
        'reviews': restaurant['reviews']
    })

# Convert to DataFrame for analysis
df_restaurants = pd.DataFrame(restaurants)

# Save scraped data
df_restaurants.to_csv('kfc_restaurant_data.csv', index=False)

# Scraping additional restaurant reviews using Selenium and BeautifulSoup
chrome_driver_path = '/path/to/chromedriver'  # Update with your driver path
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)

# Open a restaurant review page (Example)
driver.get('https://www.example-restaurant-review-site.com/restaurant-page')

# Extract reviews
soup = BeautifulSoup(driver.page_source, 'html.parser')
reviews = soup.find_all('div', class_='review')

review_data = []
for review in reviews:
    review_text = review.find('p').text
    review_rating = review.find('span', class_='rating').text
    review_data.append({
        'review_text': review_text,
        'review_rating': review_rating
    })

df_reviews = pd.DataFrame(review_data)
df_reviews.to_csv('additional_reviews.csv', index=False)

# Close Selenium driver
driver.quit()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('kfc_restaurant_data.csv')

# Preprocessing (Handle missing values, encode categorical data, etc.)
df = df.dropna()  # Drop missing values for simplicity

# Encoding categorical columns (if any)
label_enc = LabelEncoder()
df['location'] = label_enc.fit_transform(df['location'])

# Features and target variable
X = df[['location', 'rating']]  # Features (e.g., location and existing ratings)
y = df['rating']  # Target (predicted rating)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model creation
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save model (optional)
import joblib
joblib.dump(model, 'restaurant_rating_model.pkl')


from sklearn.neighbors import NearestNeighbors
import numpy as np

# Example DataFrame with restaurant latitude/longitude coordinates
df['latitude'] = np.random.uniform(-90, 90, len(df))  # Placeholder for actual latitudes
df['longitude'] = np.random.uniform(-180, 180, len(df))  # Placeholder for actual longitudes

# Prepare features for k-NN (lat/long for nearest neighbor search)
features = df[['latitude', 'longitude']].values

# Create k-NN model (finding 5 nearest neighbors)
knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(features)

# Query for a specific location
location = np.array([[40.7128, -74.0060]])  # Example (latitude/longitude for NYC)
distances, indices = knn.kneighbors(location)

# Output nearest restaurants
nearest_restaurants = df.iloc[indices[0]]
print(nearest_restaurants)


# Final dataset export for Power BI
df_final = df[['name', 'location', 'rating', 'latitude', 'longitude']]
df_final.to_csv('restaurant_data_for_powerbi.csv', index=False)

