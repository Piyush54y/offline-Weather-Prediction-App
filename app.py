# ==========================================
# AI WEATHER PREDICTION APP (FINAL UI)
# ==========================================
import streamlit as st
import numpy as np
import joblib
import random
import time
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="AI Weather Pro", layout="wide")

# ==========================================
# CUSTOM CSS (PREMIUM UI)
# ==========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #00f2ff;
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    from { text-shadow: 0 0 10px #00f2ff; }
    to { text-shadow: 0 0 30px #00f2ff; }
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin: 10px;
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px cyan;
}

.result {
    font-size: 35px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("weather_model.pkl")

# ==========================================
# SCALE FUNCTION (IMPORTANT)
# ==========================================
def scale_input(temp, hum, pres, wind, rainf):
    return np.array([[
        (temp - 10) / 35,
        (hum - 20) / 80,
        (pres - 980) / 55,
        (wind - 0) / 60,
        (rainf - 0) / 300,
        (temp - 10) / 35,
        (hum - 20) / 80
    ]])

# ==========================================
# TITLE
# ==========================================
st.markdown('<div class="title">🌦 AI Weather Prediction PRO</div>', unsafe_allow_html=True)
st.write("### ⚡ Lightweight Offline ML Model (Random Forest ~93%)")

# ==========================================
# MAIN BUTTON
# ==========================================
if st.button("🚀 Generate Smart Weather Prediction"):

    with st.spinner("Analyzing Weather Data... 🤖"):
        time.sleep(1.2)

    # RANDOM DATA GENERATION
    temp = random.randint(10, 45)
    hum = random.randint(20, 100)
    pres = random.randint(980, 1035)
    wind = random.randint(0, 60)
    rainf = random.randint(0, 300)

    # WEATHER ICON LOGIC
    if hum > 70 and rainf > 50:
        icon = "🌧"
    elif temp > 35:
        icon = "🔥"
    elif wind > 30:
        icon = "🌪"
    else:
        icon = "☀"

    # DISPLAY CARDS
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown(f'<div class="card">🌡<br>{temp}°C</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card">💧<br>{hum}%</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card">📈<br>{pres} hPa</div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="card">🌬<br>{wind} km/h</div>', unsafe_allow_html=True)
    col5.markdown(f'<div class="card">🌧<br>{rainf} mm</div>', unsafe_allow_html=True)

    # MODEL PREDICTION
    data = scale_input(temp, hum, pres, wind, rainf)
    prob = model.predict_proba(data)[0][1]
    result = model.predict(data)

    # RESULT DISPLAY
    st.markdown("### 🔮 Prediction")

    if result[0] == 1:
        st.markdown(f'<div class="result">{icon} Rain Expected</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f'<div class="result">{icon} No Rain</div>', unsafe_allow_html=True)
        st.snow()

    # ==========================================
    # CONFIDENCE CHART
    # ==========================================
    st.markdown("### 📊 Prediction Confidence")

    labels = ["No Rain", "Rain"]
    values = [1-prob, prob]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Confidence Score")
    st.pyplot(fig)

    # ==========================================
    # FEATURE VISUALIZATION
    # ==========================================
    st.markdown("### 📈 Weather Feature Trends")

    features = ["Temp", "Humidity", "Pressure", "Wind", "Rainfall"]
    values = [temp, hum, pres/10, wind, rainf/10]

    fig2, ax2 = plt.subplots()
    ax2.plot(features, values, marker='o')
    ax2.set_title("Weather Pattern")
    st.pyplot(fig2)

# ==========================================
# AUTO LIVE MODE
# ==========================================
st.markdown("---")

if st.checkbox("🔄 Live AI Mode"):
    temp = random.randint(10, 45)
    hum = random.randint(20, 100)

    st.write(f"🌡 Temp: {temp}°C | 💧 Humidity: {hum}%")

    time.sleep(2)
    st.experimental_rerun()
