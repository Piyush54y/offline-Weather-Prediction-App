# ==========================================
# 🌦 WEATHER AI PRO (ULTRA UI FINAL)
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
st.set_page_config(page_title=" ML + AQI + Weather Prediction ", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ==========================================
# 🎨 ULTRA CSS (ANIMATED + EMOJIS)
# ==========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1c92d2);
    background-size:400% 400%;
    animation: gradientMove 10s ease infinite;
    color:white;
}

@keyframes gradientMove {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* Floating Emojis */
.emoji {
    position: fixed;
    font-size: 30px;
    animation: float 10s linear infinite;
    opacity: 0.7;
}

.emoji:nth-child(1) { left:10%; animation-duration: 12s;}
.emoji:nth-child(2) { left:30%; animation-duration: 15s;}
.emoji:nth-child(3) { left:50%; animation-duration: 18s;}
.emoji:nth-child(4) { left:70%; animation-duration: 14s;}
.emoji:nth-child(5) { left:90%; animation-duration: 20s;}

@keyframes float {
    0% { top:100%; }
    100% { top:-10%; }
}

/* Title Glow */
.header {
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:#00f2ff;
    text-shadow: 0 0 20px cyan;
}

/* Big Weather */
.big {
    text-align:center;
    font-size:70px;
    font-weight:bold;
}

/* Glass Cards */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding:20px;
    border-radius:15px;
    text-align:center;
    transition:0.3s;
}

.card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px cyan;
}

/* Button */
.stButton>button {
    background: linear-gradient(45deg,#00f2ff,#0072ff);
    color:white;
    border-radius:10px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🌦 FLOATING EMOJIS
# ==========================================
st.markdown("""
<div class="emoji">☀</div>
<div class="emoji">🌧</div>
<div class="emoji">⛅</div>
<div class="emoji">🌩</div>
<div class="emoji">🔥</div>
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
    "Patna": (25.5941, 85.1376)
}

# ==========================================
# WEATHER + AQI
# ==========================================
def get_weather_aqi(lat, lon):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={lat},{lon}&aqi=yes"
    data = requests.get(url).json()

    return (
        data["current"]["temp_c"],
        data["current"]["humidity"],
        data["current"]["pressure_mb"],
        data["current"]["wind_kph"],
        data["current"].get("precip_mm", 0),
        data["current"]["air_quality"]["pm2_5"]
    )

# ==========================================
# AQI STATUS
# ==========================================
def aqi_status(aqi):
    if aqi <= 50:
        return "🟢 Good"
    elif aqi <= 100:
        return "🟡 Moderate"
    elif aqi <= 150:
        return "🟠 Unhealthy"
    elif aqi <= 200:
        return "🔴 Poor"
    else:
        return "🟣 Very Poor"

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

    # AQI STATUS COLOR
    aqi_val = aqi_status(aqi)

    if "Good" in aqi_val:
        st.success(f"🌫 Air Quality: {aqi_val}")
    elif "Moderate" in aqi_val:
        st.warning(f"🌫 Air Quality: {aqi_val}")
    else:
        st.error(f"🌫 Air Quality: {aqi_val}")

    # ML PREDICTION
    data = scale_input(temp, hum, pres, wind, rainf)
    prob = model.predict_proba(data)[0][1]
    result = model.predict(data)

    st.markdown("### 🔮 Prediction")

    if result[0] == 1:
        st.error("🌧 Rain Expected")
    else:
        st.success("☀ Clear Weather")

    st.progress(int(prob*100))
    st.write(f"Rain Probability: {prob*100:.2f}%")

    # EXPLAINABLE AI
    st.markdown("### 🧠 Why?")

    reasons = []
    if hum > 70: reasons.append("High humidity → rain chance")
    if pres < 1000: reasons.append("Low pressure → rainfall likely")
    if rainf > 0: reasons.append("Recent rainfall detected")
    if wind > 20: reasons.append("Strong wind influence")

    if len(reasons) == 0:
        st.write("✔ Weather stable")

    for r in reasons:
        st.write("✔", r)

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
st.write("⚡Weather Prediction + ML + AQI | Final Pro Version")
