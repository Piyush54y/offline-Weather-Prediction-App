# ==========================================
# 🌦 AI WEATHER PRO (ULTIMATE VERSION)
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
st.set_page_config(page_title="AI Weather Pro", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ==========================================
# CSS (ULTRA UI)
# ==========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#141E30,#243B55);
    color:white;
}
.title {
    text-align:center;
    font-size:50px;
    color:#00f2ff;
    font-weight:bold;
    animation: glow 2s infinite alternate;
}
@keyframes glow {
    from {text-shadow:0 0 10px cyan;}
    to {text-shadow:0 0 30px cyan;}
}
.card {
    background: rgba(255,255,255,0.1);
    padding:15px;
    border-radius:15px;
    text-align:center;
    transition:0.3s;
}
.card:hover {
    transform:scale(1.05);
    box-shadow:0 0 25px cyan;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================
st.markdown('<div class="title">🌦 AI Weather PRO 🌍</div>', unsafe_allow_html=True)

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
# GPS DETECTION (BROWSER BASED)
# ==========================================
st.markdown("### 📍 Detect Location")

coords = st.text_input("Enter Latitude,Longitude (Auto GPS or Google Maps)", "28.4595,77.0266")

lat, lon = map(float, coords.split(","))

# ==========================================
# WEATHER API
# ==========================================
def get_weather(lat, lon):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={lat},{lon}"
    data = requests.get(url).json()

    temp = data["current"]["temp_c"]
    hum = data["current"]["humidity"]
    pres = data["current"]["pressure_mb"]
    wind = data["current"]["wind_kph"]
    rainf = data["current"].get("precip_mm", 0)

    return temp, hum, pres, wind, rainf

# ==========================================
# MAP DISPLAY
# ==========================================
st.map(pd.DataFrame({"lat":[lat],"lon":[lon]}))

# ==========================================
# BUTTON
# ==========================================
if st.button("🚀 Predict Weather"):

    with st.spinner("🌍 Fetching Live Weather..."):
        time.sleep(1)

    temp, hum, pres, wind, rainf = get_weather(lat, lon)

    # ICON LOGIC
    if rainf > 1:
        icon = "🌧"
    elif temp > 35:
        icon = "🔥"
    elif wind > 30:
        icon = "🌪"
    else:
        icon = "☀"

    # DISPLAY CARDS
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.markdown(f'<div class="card">🌡 {temp}°C</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card">💧 {hum}%</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card">📈 {pres}</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card">🌬 {wind} km/h</div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="card">🌧 {rainf} mm</div>', unsafe_allow_html=True)

    # ML PREDICTION
    data = scale_input(temp, hum, pres, wind, rainf)
    prob = model.predict_proba(data)[0][1]
    result = model.predict(data)

    st.markdown("## 🔮 Prediction Result")

    if result[0] == 1:
        st.error(f"{icon} 🌧 Rain Expected")
        st.balloons()
    else:
        st.success(f"{icon} ☀ Clear Weather")
        st.snow()

    # CONFIDENCE BAR
    st.markdown("### 📊 Rain Probability")
    st.progress(int(prob*100))
    st.write(f"{prob*100:.2f}% chance of rain")

    # GRAPH
    st.markdown("### 📈 Weather Trend")

    features = ["Temp","Humidity","Pressure","Wind","Rain"]
    values = [temp, hum, pres/10, wind, rainf]

    fig, ax = plt.subplots()
    ax.plot(features, values, marker='o')
    ax.set_title("Live Weather Pattern")
    st.pyplot(fig)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.write("⚡ AI + ML + Real-Time Weather Prediction | Random Forest ")
