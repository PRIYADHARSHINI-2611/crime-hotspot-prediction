# 🚓 Crime Hotspot Prediction System

A Machine Learning and Flask-based system to **predict crime hotspot clusters** in a city and visualize them on an interactive map.

---

## 📌 Project Overview

This project predicts areas of high, medium, and low crime risk based on historical crime data using **K-Means clustering**.  
It also allows police or public users to **input location & time** and see the predicted risk level on a map.

---

## 🛠️ Tech Stack

- **Python** – Main programming language  
- **Flask** – Web framework for frontend  
- **Scikit-learn** – Machine Learning (K-Means clustering)  
- **Folium** – Interactive maps visualization  
- **Pandas / NumPy** – Data manipulation  
- **HTML / CSS** – Frontend interface  

---

## 🔹 Features

1. **Crime Cluster Prediction** – Predicts Low, Medium, High-risk zones  
2. **Interactive Map** – Visualizes historical crime points and predicted location  
3. **Time-based Risk Analysis** – Consider hour of the day for predictions  
4. **User Input Form** – Enter Latitude, Longitude, and Hour (0–23)  

Install dependencies:
pip install -r requirements.txt

Train the model
python train_model.py

Run the Flask app:
python app/app.py

Open your browser and go to:


🗂️ Project Structure

crime_hotspot_project/
│
├── app/
│   ├── templates/       # HTML templates
│   └── static/          # Map and static files
├── data/                # Dataset (CSV)
├── models/              # Saved ML model & scaler
├── src/                 # Preprocessing & model scripts
├── train_model.py       # Model training script
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

