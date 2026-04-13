# ==========================================
# 🌦 AI WEATHER PRO (FINAL PRO VERSION)
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
# CSS
# ==========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#141E30,#243B55);
    color:white;
}
.title {
    text-align:center;
    font-size:45px;
    color:#00f2ff;
    font-weight:bold;
}
.card {
    background: rgba(255,255,255,0.1);
    padding:15px;
    border-radius:12px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌦 AI Weather Prediction PRO</div>', unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("weather_model.pkl")

# ==========================================
# SCALE
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

    days = []
    temps = []

    for d in data["forecast"]["forecastday"]:
        days.append(d["date"])
        temps.append(d["day"]["avgtemp_c"])

    return days, temps

# ==========================================
# UI TABS
# ==========================================
tab1, tab2 = st.tabs(["🌦 Prediction", "📊 Forecast"])

# ==========================================
# TAB 1: PREDICTION
# ==========================================
with tab1:

    st.markdown("### 📍 Select City")
    city = st.selectbox("Choose city:", list(city_coords.keys()))
    lat, lon = city_coords[city]

    if st.button("🚀 Get Prediction"):

        temp, hum, pres, wind, rainf = get_weather(lat, lon)

        # ICON
        if rainf > 1:
            icon = "🌧"
        elif temp > 35:
            icon = "🔥"
        else:
            icon = "☀"

        # CARDS
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("🌡 Temp", f"{temp}°C")
        c2.metric("💧 Humidity", f"{hum}%")
        c3.metric("📈 Pressure", pres)
        c4.metric("🌬 Wind", wind)
        c5.metric("🌧 Rainfall", rainf)

        # ML
        data = scale_input(temp, hum, pres, wind, rainf)
        prob = model.predict_proba(data)[0][1]
        result = model.predict(data)

        st.markdown("## 🔮 Prediction")

        if result[0] == 1:
            st.error(f"{icon} Rain Expected")
        else:
            st.success(f"{icon} Clear Weather")

        # CONFIDENCE
        st.progress(int(prob*100))
        st.write(f"{prob*100:.2f}% probability")

        # EXPLAINABLE AI
        st.markdown("### 🧠 Why?")
        if hum > 70:
            st.write("✔ High humidity increases rain chance")
        if pres < 1000:
            st.write("✔ Low pressure supports rainfall")

        # DOWNLOAD
        df = pd.DataFrame({
            "Temp":[temp],"Humidity":[hum],
            "Prediction":[result[0]]
        })

        st.download_button("📥 Download Report", df.to_csv(), "report.csv")

# ==========================================
# TAB 2: FORECAST
# ==========================================
with tab2:

    st.markdown("### 📅 7-Day Forecast")

    city = st.selectbox("Select city for forecast:", list(city_coords.keys()), key="forecast")
    lat, lon = city_coords[city]

    days, temps = get_forecast(lat, lon)

    # CHART
    st.line_chart(temps)

    # TABLE
    forecast_df = pd.DataFrame({
        "Day": days,
        "Avg Temp": temps
    })

    st.dataframe(forecast_df)

# ==========================================
# LIVE MODE
# ==========================================
st.markdown("---")

if st.checkbox("🔄 Live Mode"):
    time.sleep(3)
    st.rerun()

# ==========================================
# FOOTER
# ==========================================
st.write("⚡ ML + Real-Time Weather | Random Forest (~93%)")
