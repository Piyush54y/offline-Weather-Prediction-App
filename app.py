# ==========================================
# AI WEATHER APP (FINAL WITH RESULTS + BEST MODEL)
# ==========================================
import streamlit as st
import numpy as np
import joblib
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="AI Weather Pro", layout="wide")

# ==========================================
# CSS
# ==========================================
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);color:white;}
.title {text-align:center;font-size:40px;color:cyan;font-weight:bold;}
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
# GENERATE DATASET (FOR MODEL COMPARISON)
# ==========================================
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 3000

    temp = np.random.uniform(10,45,n)
    hum = np.random.uniform(20,100,n)
    pres = np.random.uniform(980,1035,n)
    wind = np.random.uniform(0,60,n)
    rainf = np.random.uniform(0,300,n)

    score = (0.65*(hum/100) + 0.25*(rainf/300) +
             0.07*(1-(pres-980)/55) + 0.03*(temp/45))

    rain = np.where(score > 0.58, 1, 0)

    df = pd.DataFrame({
        "Temp": temp, "Hum": hum, "Pres": pres,
        "Wind": wind, "Rainf": rainf, "Rain": rain
    })

    X = df.drop("Rain", axis=1)
    y = df["Rain"]

    return X, y

X, y = generate_data()

# ==========================================
# TRAIN MODELS (FOR COMPARISON)
# ==========================================
@st.cache_resource
def train_models(X, y):
    from sklearn.model_selection import train_test_split

    X = np.array(X)
    y = np.array(y)

    X = np.hstack((X, X[:,[0]], X[:,[1]]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=8),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=150)
    }

    results = []
    best_model = None
    best_acc = 0

    for name, m in models.items():
        start = time.time()
        m.fit(X_train, y_train)
        end = time.time()

        pred = m.predict(X_test)
        acc = accuracy_score(y_test, pred)*100

        results.append([name, acc, end-start])

        if acc > best_acc:
            best_acc = acc
            best_model = name

    df_res = pd.DataFrame(results, columns=["Model","Accuracy (%)","Time (s)"])

    return df_res, best_model, best_acc

results_df, best_model_name, best_acc = train_models(X, y)

# ==========================================
# SIDEBAR MODE
# ==========================================
mode = st.sidebar.radio("Select Mode", [
    "🔮 Prediction",
    "📊 Model Results",
    "🔄 Live Mode"
])

# ==========================================
# MODE 1: PREDICTION
# ==========================================
if mode == "🔮 Prediction":

    if st.button("🚀 Generate Prediction"):

        temp = random.randint(10,45)
        hum = random.randint(20,100)
        pres = random.randint(980,1035)
        wind = random.randint(0,60)
        rainf = random.randint(0,300)

        st.write(f"🌡 Temp: {temp}°C | 💧 Humidity: {hum}%")

        data = scale_input(temp, hum, pres, wind, rainf)
        prob = model.predict_proba(data)[0][1]
        result = model.predict(data)

        if result[0] == 1:
            st.error("🌧 Rain Expected")
        else:
            st.success("☀ No Rain")

        st.progress(int(prob*100))
        st.write(f"Confidence: {prob*100:.2f}%")

# ==========================================
# MODE 2: RESULTS TABLE ⭐
# ==========================================
elif mode == "📊 Model Results":

    st.markdown("## 📊 Model Comparison Table")

    st.dataframe(results_df)

    st.markdown("### 🏆 Best Model")
    st.success(f"{best_model_name} with Accuracy = {best_acc:.2f}%")

    # Graph
    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["Accuracy (%)"])
    ax.set_title("Accuracy Comparison")
    st.pyplot(fig)

# ==========================================
# MODE 3: LIVE MODE (FIXED)
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
    st.rerun()
