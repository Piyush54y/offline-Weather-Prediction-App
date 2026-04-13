# ==========================================
# 🌦 AI WEATHER PRO (FINAL STABLE VERSION)
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
# CSS (CLEAN + ANIMATED)
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
    font-weight:bold;
    color:#00f2ff;
    animation: glow 2s infinite alternate;
}
@keyframes glow {
    from {text-shadow:0 0 10px #00f2ff;}
    to {text-shadow:0 0 25px #00f2ff;}
}
.card {
    background: rgba(255,255,255,0.08);
    padding:15px;
    border-radius:10px;
    text-align:center;
    margin:5px;
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
# AUTO DETECT (FALLBACK ONLY)
# ==========================================
def detect_city():
    try:
        res = requests.get("http://ip-api.com/json").json()
        return res.get("city", "Delhi")
    except:
        return "Delhi"

# ==========================================
# WEATHER API
# ==========================================
def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    data = requests.get(url).json()

    temp = data["current"]["temp_c"]
    hum = data["current"]["humidity"]
    pres = data["current"]["pressure_mb"]
    wind = data["current"]["wind_kph"]
    rainf = data["current"].get("precip_mm", 0)

    return temp, hum, pres, wind, rainf

# ==========================================
# LOCATION SELECTION (FIXED 🔥)
# ==========================================
st.markdown("### 📍 Select Your Location")

auto_city = detect_city()

popular_cities = [
    "Gurugram", "Delhi", "Mumbai", "Bangalore",
    "Patna", "Kolkata", "Hyderabad"
]

city = st.selectbox(
    "Choose city (or type manually below):",
    popular_cities
)

custom_city = st.text_input("Or enter your city manually:", value=city)

if custom_city:
    city = custom_city

st.write(f"📍 Using Location: **{city}**")

# ==========================================
# MAIN BUTTON
# ==========================================
if st.button("🚀 Get Live Prediction"):

    with st.spinner("Fetching Live Weather... 🌍"):
        time.sleep(1)

    temp, hum, pres, wind, rainf = get_weather(city)

    # ICON LOGIC
    if hum > 70 and rainf > 1:
        icon = "🌧"
    elif temp > 35:
        icon = "🔥"
    else:
        icon = "☀"

    # DISPLAY CARDS
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
st.write("⚡ Powered by ML + Real-Time API | Random Forest ")
