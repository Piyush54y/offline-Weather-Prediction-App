# ==========================================
# 🌦 AI WEATHER PRO (AQI + PREDICTION)
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
# CSS (PREMIUM UI)
# ==========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1c92d2);
    background-size:400% 400%;
    animation: gradientBG 12s ease infinite;
    color:white;
}
@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}
.header {
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#00f2ff;
}
.big {
    text-align:center;
    font-size:60px;
    font-weight:bold;
}
.card {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    padding:20px;
    border-radius:15px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

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
    "Patna": (25.5941, 85.1376)
}

# ==========================================
# WEATHER + AQI API
# ==========================================
def get_weather_aqi(lat, lon):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={lat},{lon}&aqi=yes"
    data = requests.get(url).json()

    temp = data["current"]["temp_c"]
    hum = data["current"]["humidity"]
    pres = data["current"]["pressure_mb"]
    wind = data["current"]["wind_kph"]
    rainf = data["current"].get("precip_mm", 0)

    aqi = data["current"]["air_quality"]["pm2_5"]

    return temp, hum, pres, wind, rainf, aqi

# ==========================================
# AQI STATUS
# ==========================================
def aqi_status(aqi):
    if aqi <= 50:
        return "🟢 Good"
    elif aqi <= 100:
        return "🟡 Moderate"
    else:
        return "🔴 Poor"

# ==========================================
# CITY SELECT
# ==========================================
city = st.selectbox("📍 Select City", list(city_coords.keys()))
lat, lon = city_coords[city]

# ==========================================
# BUTTON
# ==========================================
if st.button("🚀 Get Weather"):

    temp, hum, pres, wind, rainf, aqi = get_weather_aqi(lat, lon)

    # ICON
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
    c1,c2,c3,c4,c5,c6 = st.columns(6)

    c1.markdown(f'<div class="card">🌡<br>{temp}°C</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card">💧<br>{hum}%</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card">📈<br>{pres}</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card">🌬<br>{wind}</div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="card">🌧<br>{rainf}</div>', unsafe_allow_html=True)
    c6.markdown(f'<div class="card">🌫 AQI<br>{aqi:.1f}</div>', unsafe_allow_html=True)

    # AQI STATUS
    st.markdown(f"### 🌫 Air Quality: {aqi_status(aqi)}")

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
    st.markdown("### 🧠 Why?")
    if hum > 70:
        st.write("✔ High humidity → rain chance")
    if pres < 1000:
        st.write("✔ Low pressure → rainfall likely")

    # GRAPH
    st.markdown("### 📊 Weather Insights")

    fig, ax = plt.subplots()
    ax.plot(["Temp","Humidity","Pressure","Wind","Rain","AQI"],
            [temp, hum, pres/10, wind, rainf, aqi], marker='o')
    st.pyplot(fig)

# ==========================================
# LIVE MODE
# ==========================================
if st.checkbox("🔄 Live Mode"):
    time.sleep(3)
    st.rerun()

# ==========================================
# FOOTER
# ==========================================
st.write("⚡ ML + Real-Time Weather + AQI | Premium UI")
