# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

def load_data(file_path):
    """
    Load CSV data, drop missing values, and extract hour/day features.
    """
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    
    # Extract Hour and Day from Date
    df["Hour"] = pd.to_datetime(df["Date"]).dt.hour
    df["Day"] = pd.to_datetime(df["Date"]).dt.day
    
    return df

def scale_features(df, feature_cols):
    """
    Scale selected features using MinMaxScaler and save scaler.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    
    # Save the scaler for later use
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    
    return scaled
