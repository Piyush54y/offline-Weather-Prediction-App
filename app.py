import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import pydeck as pdk
import time

st.set_page_config(page_title="Weather AI Pro", page_icon="🌦", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

st.markdown("""
<style>
.stApp{
background:linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1c92d2);
background-size:400% 400%;
animation:gradientMove 12s ease infinite;
}
@keyframes gradientMove{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_weather(city):
    url=f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=yes"
    return requests.get(url,timeout=10).json()

@st.cache_data(ttl=300)
def get_forecast(city):
    url=f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={city}&days=7&aqi=yes"
    return requests.get(url,timeout=10).json()

st.title("🌦 Weather AI Pro")

city=st.text_input("Enter City","Delhi")

if city:
    data=get_weather(city)

    if "error" in data:
        st.error(data["error"]["message"])
    else:
        loc=data["location"]
        cur=data["current"]

        st.metric("Temperature", f"{cur['temp_c']} °C")
        st.metric("Humidity", f"{cur['humidity']} %")
        st.metric("Pressure", f"{cur['pressure_mb']} mb")

        aqi=cur["air_quality"]["pm2_5"]
        st.metric("PM2.5 AQI", round(aqi,2))

        st.subheader("🗺 Map")
        df=pd.DataFrame({"lat":[loc["lat"]],"lon":[loc["lon"]]})
        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=loc["lat"],
                    longitude=loc["lon"],
                    zoom=8
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=df,
                        get_position='[lon, lat]',
                        get_radius=5000,
                    )
                ]
            )
        )

        st.subheader("📅 7 Day Forecast")
        forecast=get_forecast(city)

        for d in forecast["forecast"]["forecastday"]:
            st.write(
                f"{d['date']} | Temp: {d['day']['avgtemp_c']}°C | Rain Chance: {d['day']['daily_chance_of_rain']}%"
            )

        fig, ax = plt.subplots()
        ax.bar(
            ["Temp","Humidity","Pressure"],
            [cur["temp_c"], cur["humidity"], cur["pressure_mb"]/10]
        )
        st.pyplot(fig)

        report=f"""
City: {loc['name']}
Country: {loc['country']}
Temperature: {cur['temp_c']} C
Humidity: {cur['humidity']}%
AQI PM2.5: {aqi}
"""
        st.download_button("Download Report", report, file_name="weather_report.txt")

if st.checkbox("Live Mode"):
    time.sleep(60)
    st.rerun()
