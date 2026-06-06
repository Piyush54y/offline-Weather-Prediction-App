
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import time

st.set_page_config(page_title="WeatherVerse AI Ultimate", layout="wide")
.stApp{
background:#0f172a;
}

.metric-card{
background:#1e293b;
padding:20px;
border-radius:15px;
box-shadow:0 4px 20px rgba(0,0,0,.2);
}

h1{
text-align:center;
}
st.markdown('''
<style>
.stApp{
background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}
.metric-card{
padding:15px;border-radius:15px;
background:rgba(255,255,255,0.08);
}
</style>
''', unsafe_allow_html=True)

API_KEY="efd7a881ace6419480e100155251006"

st.title("🌦 WeatherVerse AI Ultimate")

city=st.text_input("Enter City","Delhi")

def get_data(city):
    url=f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={city}&days=7&aqi=yes"
    return requests.get(url).json()

if city:
    data=get_data(city)
    if "error" not in data:
        cur=data["current"]
        loc=data["location"]

        c1,c2,c3,c4=st.columns(4)
        c1.metric("🌡 Temp", f"{cur['temp_c']}°C")
        c2.metric("💧 Humidity", f"{cur['humidity']}%")
        c3.metric("🌬 Wind", f"{cur['wind_kph']} kph")
        c4.metric("🌫 PM2.5", round(cur["air_quality"]["pm2_5"],2))

        st.subheader("🗺 Map")
        df=pd.DataFrame({"lat":[loc["lat"]],"lon":[loc["lon"]]})
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=loc["lat"],
                longitude=loc["lon"],
                zoom=8,pitch=50),
            layers=[pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position='[lon, lat]',
                get_radius=5000)]
        ))

        st.subheader("📅 Forecast")
        days=data["forecast"]["forecastday"]
        chart_df=pd.DataFrame({
            "Date":[d["date"] for d in days],
            "Temp":[d["day"]["avgtemp_c"] for d in days]
        })
        st.plotly_chart(px.line(chart_df,x="Date",y="Temp"))

        st.subheader("🎯 AQI Gauge")
        fig=go.Figure(go.Indicator(
            mode="gauge+number",
            value=cur["air_quality"]["pm2_5"],
            title={"text":"PM2.5"}))
        st.plotly_chart(fig)

        st.subheader("📥 Report")
        report=f"City: {loc['name']}\nTemp:{cur['temp_c']}"
        st.download_button("Download Report",report,"weather_report.txt")

if st.checkbox("🔄 Auto Refresh"):
    time.sleep(60)
    st.rerun()
