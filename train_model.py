# train_model.py
from src.preprocessing import load_data, scale_features
from src.model import train_kmeans

# Load and preprocess data
df = load_data("data/crime_data.csv")

# Select features for clustering
features = ["Latitude", "Longitude", "Hour"]
X_scaled = scale_features(df, features)

# Train KMeans model
kmeans = train_kmeans(X_scaled, n_clusters=3)

print("✅ Model and scaler trained and saved successfully!")
