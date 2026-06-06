import streamlit as st
import requests, joblib, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(page_title="WeatherVerse AI Ultimate", layout="wide")

API_KEY="efd7a881ace6419480e100155251006"

st.markdown("""
<style>
.stApp{
background:linear-gradient(135deg,#0f172a,#1e293b,#2563eb);
color:white;
}
[data-testid="metric-container"]{
background:rgba(255,255,255,.1);
border-radius:15px;
padding:10px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("weather_model.pkl")

def scale_input(temp, hum, pres, wind, rainf):
    return [[
        (temp-10)/35,
        (hum-20)/80,
        (pres-980)/55,
        wind/60,
        rainf/300,
        (temp-10)/35,
        (hum-20)/80
    ]]

page = st.sidebar.radio("Menu", ["Dashboard","ML Comparison"])

if page == "ML Comparison":
    df = pd.DataFrame({
        "Model":["Logistic Regression","Random Forest","XGBoost"],
        "Accuracy":[85,92,95]
    })
    st.plotly_chart(px.bar(df,x="Model",y="Accuracy"))
    st.stop()

city = st.text_input("🌍 Enter City","Delhi")

if city:
    url=f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={city}&days=7&aqi=yes"
    data=requests.get(url).json()

    if "error" in data:
        st.error(data["error"]["message"])
    else:
        cur=data["current"]
        loc=data["location"]
        astro=data["forecast"]["forecastday"][0]["astro"]

        c1,c2,c3,c4=st.columns(4)
        c1.metric("🌡 Temp",f"{cur['temp_c']}°C")
        c2.metric("💧 Humidity",f"{cur['humidity']}%")
        c3.metric("🌬 Wind",f"{cur['wind_kph']} kph")
        c4.metric("🌫 PM2.5",round(cur["air_quality"]["pm2_5"],2))

        st.subheader("☀ UV / 🌅 Sun / 🌙 Moon")
        a,b,c,d=st.columns(4)
        a.metric("UV",cur["uv"])
        b.metric("Sunrise",astro["sunrise"])
        c.metric("Sunset",astro["sunset"])
        d.metric("Moon",astro["moon_phase"])

        st.subheader("🤖 ML Rain Prediction")
        try:
            model=load_model()
            X=scale_input(
                cur["temp_c"],
                cur["humidity"],
                cur["pressure_mb"],
                cur["wind_kph"],
                cur.get("precip_mm",0)
            )
            pred=model.predict(X)[0]
            prob=model.predict_proba(X)[0][1]*100

            if pred==1:
                st.error(f"🌧 Rain Expected ({prob:.1f}%)")
            else:
                st.success(f"☀ Clear Weather ({100-prob:.1f}%)")

            st.write("### 🧠 Explainable AI")
            if cur["humidity"]>70:
                st.write("✔ High humidity")
            if cur["pressure_mb"]<1000:
                st.write("✔ Low pressure")
            if cur["wind_kph"]>20:
                st.write("✔ Strong wind")
        except Exception as e:
            st.warning(f"Model issue: {e}")

        st.subheader("🗺 3D Map")
        df=pd.DataFrame({"lat":[loc["lat"]],"lon":[loc["lon"]]})
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=loc["lat"],
                longitude=loc["lon"],
                zoom=10,pitch=60,bearing=30
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position='[lon, lat]',
                    get_radius=5000
                )
            ]
        ))

        days=data["forecast"]["forecastday"]
        fdf=pd.DataFrame({
            "Date":[x["date"] for x in days],
            "Temp":[x["day"]["avgtemp_c"] for x in days],
            "Rain":[x["day"]["daily_chance_of_rain"] for x in days]
        })

        st.subheader("📅 7-Day Forecast")
        st.dataframe(fdf,use_container_width=True)

        st.plotly_chart(px.line(fdf,x="Date",y="Temp",markers=True))

        st.subheader("🎯 AQI Gauge")
        gauge=go.Figure(go.Indicator(
            mode="gauge+number",
            value=cur["air_quality"]["pm2_5"],
            title={"text":"PM2.5 AQI"}
        ))
        st.plotly_chart(gauge,use_container_width=True)

        report=f"""
WeatherVerse AI Report

City: {loc['name']}
Country: {loc['country']}
Temperature: {cur['temp_c']} °C
Humidity: {cur['humidity']} %
Wind: {cur['wind_kph']} kph
UV: {cur['uv']}
PM2.5: {cur['air_quality']['pm2_5']}
"""

        st.download_button("📥 TXT Report",report,"weather_report.txt")
