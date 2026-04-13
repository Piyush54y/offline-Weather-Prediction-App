# ==========================================
# 🌦 AI WEATHER PRO (LAT-LONG ACCURATE VERSION)
# ==========================================
import streamlit as st
import numpy as np
import joblib
import requests
import time
import matplotlib.pyplot as plt

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="AI Weather Pro", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ==========================================
# CSS (PREMIUM UI)
# ==========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
}
.title {
    text-align:center;
    font-size:40px;
    color:#00f2ff;
    font-weight:bold;
}
.card {
    background: rgba(255,255,255,0.08);
    padding:15px;
    border-radius:10px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================
st.markdown('<div class="title">🌦 AI Weather Prediction PRO</div>', unsafe_allow_html=True)

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
# CITY → LAT LONG MAP (ACCURATE)
# ==========================================
city_coords = {
    "Gurugram": (28.4595, 77.0266),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Patna": (25.5941, 85.1376)
}

# ==========================================
# WEATHER FUNCTION (LAT-LONG)
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
# LOCATION SELECT
# ==========================================
st.markdown("### 📍 Select Location")

city = st.selectbox("Choose City", list(city_coords.keys()))

lat, lon = city_coords[city]

st.write(f"📍 Using: **{city}** (Lat: {lat}, Lon: {lon})")

# ==========================================
# BUTTON
# ==========================================
if st.button("🚀 Get Accurate Prediction"):

    with st.spinner("Fetching Accurate Weather... 🌍"):
        time.sleep(1)

    temp, hum, pres, wind, rainf = get_weather(lat, lon)

    # ICON
    if hum > 70 and rainf > 1:
        icon = "🌧"
    elif temp > 35:
        icon = "🔥"
    else:
        icon = "☀"

    # CARDS
    c1, c2, c3, c4, c5 = st.columns(5)

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
        st.error(f"{icon} Rain Expected")
        st.balloons()
    else:
        st.success(f"{icon} No Rain")
        st.snow()

    # CONFIDENCE
    st.markdown("### 📊 Confidence")
    st.progress(int(prob*100))
    st.write(f"{prob*100:.2f}% probability of rain")

    # GRAPH
    st.markdown("### 📈 Weather Pattern")

    features = ["Temp","Humidity","Pressure","Wind","Rain"]
    values = [temp, hum, pres/10, wind, rainf]

    fig, ax = plt.subplots()
    ax.plot(features, values, marker='o')
    ax.set_title("Weather Trend")
    st.pyplot(fig)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.write("⚡ Accurate Location-Based ML Prediction | Random Forest (~93%)")
