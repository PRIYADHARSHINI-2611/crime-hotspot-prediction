from flask import Flask, render_template, request
import pandas as pd
import folium
import numpy as np
import pickle
import os

# ---------------- BASE DIR ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- FLASK APP ----------------
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# ---------------- LOAD MODEL ----------------
scaler = pickle.load(open(os.path.join(BASE_DIR, "..", "models", "scaler.pkl"), "rb"))
kmeans = pickle.load(open(os.path.join(BASE_DIR, "..", "models", "kmeans_model.pkl"), "rb"))

# ---------------- LOAD DATA ----------------
df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "crime_data.csv"))

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    cluster_result = None
    risk_message = ""
    risk_color = ""
    lat = lon = None

    if request.method == "POST":
        lat = float(request.form["latitude"])
        lon = float(request.form["longitude"])
        hour = int(request.form["hour"])

        # Predict cluster
        new_point = np.array([[lat, lon, hour]])
        new_scaled = scaler.transform(new_point)
        cluster_result = int(kmeans.predict(new_scaled)[0])

        # Time-based risk
        if hour >= 20 or hour <= 5:
            risk_message = "🚨 HIGH RISK: Crime-prone night hours"
            risk_color = "red"
        elif 6 <= hour <= 18:
            risk_message = "⚠️ MODERATE RISK: Day time"
            risk_color = "orange"
        else:
            risk_message = "✅ LOW RISK: Safer hours"
            risk_color = "green"

    # ---------------- MAP ----------------
    m = folium.Map(
        location=[df["Latitude"].mean(), df["Longitude"].mean()],
        zoom_start=12
    )

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color="red",
            fill=True,
            fill_opacity=0.6,
            popup=row["Crime_Type"]
        ).add_to(m)

    if lat is not None:
        folium.Marker(
            location=[lat, lon],
            popup=f"Predicted Cluster: {cluster_result}",
            icon=folium.Icon(color="blue")
        ).add_to(m)

    # ---------------- SAVE MAP SAFELY ----------------
    static_dir = os.path.join(BASE_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)

    map_path = os.path.join(static_dir, "map.html")
    m.save(map_path)

    return render_template(
        "index.html",
        cluster_result=cluster_result,
        risk_message=risk_message,
        risk_color=risk_color
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)
