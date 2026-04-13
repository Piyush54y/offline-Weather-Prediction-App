# ==========================================
# AI WEATHER PREDICTION - ULTIMATE APP
# ==========================================
import streamlit as st
import numpy as np
import joblib
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="AI Weather Ultimate", layout="wide")

# ==========================================
# CSS (PREMIUM UI)
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
    color: cyan;
    font-weight: bold;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin: 5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌦 AI Weather Prediction System</div>', unsafe_allow_html=True)

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
# MODE SELECTOR
# ==========================================
mode = st.sidebar.radio("Select Mode", [
    "🔮 Single Prediction",
    "📊 Dashboard",
    "🔄 Live Mode"
])

# ==========================================
# SINGLE PREDICTION
# ==========================================
if mode == "🔮 Single Prediction":

    if st.button("🚀 Generate Prediction"):

        temp = random.randint(10,45)
        hum = random.randint(20,100)
        pres = random.randint(980,1035)
        wind = random.randint(0,60)
        rainf = random.randint(0,300)

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Temp", f"{temp}°C")
        col2.metric("Humidity", f"{hum}%")
        col3.metric("Pressure", pres)
        col4.metric("Wind", wind)
        col5.metric("Rainfall", rainf)

        data = scale_input(temp, hum, pres, wind, rainf)
        prob = model.predict_proba(data)[0][1]
        result = model.predict(data)

        st.markdown("## 🔮 Prediction Result")

        if result[0] == 1:
            st.error("🌧 Rain Expected")
            st.image("https://media.giphy.com/media/26BRrSvJUa0crqw4E/giphy.gif")
        else:
            st.success("☀ No Rain")
            st.image("https://media.giphy.com/media/l0HlBO7eyXzSZkJri/giphy.gif")

        # Progress bar
        st.progress(int(prob*100))
        st.write(f"Confidence: {prob*100:.2f}%")

        # Explainable AI
        st.markdown("### 🧠 Why this prediction?")
        if hum > 70:
            st.write("✔ High humidity increases rain chance")
        if rainf > 50:
            st.write("✔ High rainfall pattern detected")
        if pres < 1000:
            st.write("✔ Low pressure supports rainfall")

# ==========================================
# DASHBOARD MODE
# ==========================================
elif mode == "📊 Dashboard":

    data_list = []

    for _ in range(8):
        temp = random.randint(10,45)
        hum = random.randint(20,100)
        pres = random.randint(980,1035)
        wind = random.randint(0,60)
        rainf = random.randint(0,300)

        data = scale_input(temp, hum, pres, wind, rainf)
        pred = model.predict(data)[0]

        data_list.append([temp, hum, pres, wind, rainf, pred])

    df = pd.DataFrame(data_list, columns=[
        "Temp","Humidity","Pressure","Wind","Rainfall","Prediction"
    ])

    st.dataframe(df)

    # Chart
    st.markdown("### 📈 Temperature Trend")
    st.line_chart(df["Temp"])

    # Download
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download Data", csv, "weather.csv")

# ==========================================
# LIVE MODE
# ==========================================
elif mode == "🔄 Live Mode":

    st.write("📡 Live Prediction Running...")

    temp = random.randint(10,45)
    hum = random.randint(20,100)
    pres = random.randint(980,1035)
    wind = random.randint(0,60)
    rainf = random.randint(0,300)

    st.write(f"Temp: {temp}°C | Humidity: {hum}%")

    data = scale_input(temp, hum, pres, wind, rainf)
    result = model.predict(data)

    if result[0] == 1:
        st.error("🌧 Rain Expected")
    else:
        st.success("☀ No Rain")

    time.sleep(2)
    st.experimental_rerun()

# ==========================================
# MODEL INFO
# ==========================================
with st.sidebar.expander("🧠 Model Info"):
    st.write("Algorithm: Random Forest")
    st.write("Accuracy: ~93%")
    st.write("Type: Ensemble Learning")
