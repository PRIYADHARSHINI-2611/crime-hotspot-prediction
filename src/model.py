# src/model.py
from sklearn.cluster import KMeans
import pickle

def train_kmeans(X, n_clusters=3):
    """
    Train KMeans clustering and save the model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    
    # Save trained model
    pickle.dump(kmeans, open("models/kmeans_model.pkl", "wb"))
    
    return kmeans

def predict_cluster(new_data):
    """
    Predict cluster for new input using saved scaler and KMeans model.
    """
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    kmeans = pickle.load(open("models/kmeans_model.pkl", "rb"))
    
    # Scale new input
    scaled_data = scaler.transform(new_data)
    
    # Predict cluster
    cluster = kmeans.predict(scaled_data)
    
    return cluster
