# ==========================================
# 🌦 AI WEATHER ULTRA PRO (MOBILE UI STYLE)
# ==========================================
import streamlit as st
import numpy as np
import joblib
import requests
import time
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Weather Pro", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ==========================================
# ULTRA CSS (MOBILE APP STYLE 🔥)
# ==========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1c92d2);
    background-size:400% 400%;
    animation: gradientBG 12s ease infinite;
    color:white;
}

/* Background animation */
@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* Header */
.header {
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:#00f2ff;
    text-shadow:0 0 20px cyan;
}

/* Big weather display */
.big {
    text-align:center;
    font-size:60px;
    font-weight:bold;
}

/* Glass cards */
.card {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    padding:20px;
    border-radius:20px;
    text-align:center;
    transition:0.3s;
}

.card:hover {
    transform:scale(1.05);
    box-shadow:0 0 25px cyan;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(45deg,#00f2ff,#0072ff);
    color:white;
    border-radius:10px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================
st.markdown('<div class="header">🌦 Weather AI PRO</div>', unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("weather_model.pkl")

# ==========================================
# SCALE FUNCTION
# ==========================================
def scale_input(temp, hum, pres, wind, rainf):
    return np.array([[
        (temp - 10)/35,
        (hum - 20)/80,
        (pres - 980)/55,
        wind/60,
        rainf/300,
        (temp - 10)/35,
        (hum - 20)/80
    ]])

# ==========================================
# CITY DATA
# ==========================================
city_coords = {
    "Gurugram": (28.4595, 77.0266),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Patna": (25.5941, 85.1376),
    "Kolkata": (22.5726, 88.3639)
}

# ==========================================
# WEATHER API
# ==========================================
def get_weather(lat, lon):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={lat},{lon}"
    data = requests.get(url).json()

    return (
        data["current"]["temp_c"],
        data["current"]["humidity"],
        data["current"]["pressure_mb"],
        data["current"]["wind_kph"],
        data["current"].get("precip_mm", 0)
    )

# ==========================================
# FORECAST
# ==========================================
def get_forecast(lat, lon):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={lat},{lon}&days=7"
    data = requests.get(url).json()

    days, temps = [], []

    for d in data["forecast"]["forecastday"]:
        days.append(d["date"])
        temps.append(d["day"]["avgtemp_c"])

    return days, temps

# ==========================================
# CITY SELECT
# ==========================================
city = st.selectbox("📍 Select City", list(city_coords.keys()))
lat, lon = city_coords[city]

# ==========================================
# MAIN BUTTON
# ==========================================
if st.button("🚀 Get Weather"):

    temp, hum, pres, wind, rainf = get_weather(lat, lon)

    # ICON + STATUS
    if rainf > 1:
        icon = "🌧"
        status = "Rainy"
    elif temp > 35:
        icon = "🔥"
        status = "Hot"
    else:
        icon = "☀"
        status = "Clear"

    # BIG DISPLAY
    st.markdown(f"""
    <div class="big">{icon} {temp}°C</div>
    <h3 style="text-align:center;">{status} Weather</h3>
    """, unsafe_allow_html=True)

    # CARDS
    c1,c2,c3,c4,c5 = st.columns(5)

    c1.markdown(f'<div class="card">🌡<br>{temp}°C</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card">💧<br>{hum}%</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card">📈<br>{pres}</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card">🌬<br>{wind}</div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="card">🌧<br>{rainf}</div>', unsafe_allow_html=True)

    # ML PREDICTION
    data = scale_input(temp, hum, pres, wind, rainf)
    prob = model.predict_proba(data)[0][1]
    result = model.predict(data)

    st.markdown("### 🔮 Prediction")

    if result[0] == 1:
        st.error("🌧 Rain Expected")
    else:
        st.success("☀ Clear Weather")

    # CONFIDENCE
    st.progress(int(prob*100))
    st.write(f"Rain Probability: {prob*100:.2f}%")

    # EXPLAINABLE AI
    st.markdown("### 🧠 Why Prediction?")
    if hum > 70:
        st.write("✔ High humidity → rain chance")
    if pres < 1000:
        st.write("✔ Low pressure → rainfall likely")

    # FORECAST
    st.markdown("### 📅 7-Day Forecast")

    days, temps = get_forecast(lat, lon)
    st.line_chart(temps)

    # GRAPH
    st.markdown("### 📊 Weather Insights")

    fig, ax = plt.subplots()
    ax.plot(["Temp","Humidity","Pressure","Wind","Rain"],
            [temp, hum, pres/10, wind, rainf], marker='o')
    st.pyplot(fig)

    # DOWNLOAD
    df = pd.DataFrame({
        "Temp":[temp],
        "Humidity":[hum],
        "Prediction":[result[0]]
    })

    st.download_button("📥 Download Report", df.to_csv(), "weather.csv")

# ==========================================
# LIVE MODE
# ==========================================
if st.checkbox("🔄 Live Mode"):
    time.sleep(3)
    st.rerun()

# ==========================================
# FOOTER
# ==========================================
st.write("⚡ AI + ML + Real-Time Weather | Premium UI")
