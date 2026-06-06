# Weather Prediction Using Machine Learning
This project predicts weather conditions using Machine Learning algorithms and historical weather data. The system analyzes parameters such as temperature, humidity, and wind speed to forecast future weather. A Streamlit web interface allows users to enter data and view predictions instantly. The model is trained using supervised learning techniques. The project demonstrates the practical application of Machine Learning in weather forecasting.
## Overview
This project predicts weather conditions using Machine Learning and historical weather data.

## Features
- Temperature Prediction
- Humidity Analysis
- Weather Forecasting
- Streamlit Dashboard
- Data Visualization

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib

## Installation

pip install -r requirements.txt

3. Source Code Explanation (Short)
Import Libraries
import pandas as pd

Used to load and manage datasets.

from sklearn.model_selection import train_test_split

Used to divide data into training and testing sets.

from sklearn.ensemble import RandomForestRegressor

Imports Random Forest Machine Learning algorithm.

import streamlit as st

Creates the web interface.

Load Dataset
df = pd.read_csv("weather.csv")

Loads weather data from CSV file.

Features and Target
X = df[['Humidity','Wind']]
y = df['Temperature']

Input variables and output variable.

Train Model
model = RandomForestRegressor()
model.fit(X,y)

Trains the model using historical data.

Prediction
prediction = model.predict([[70,10]])

Predicts temperature for given humidity and wind speed.

Display Result
st.write(prediction)

Shows output on Streamlit webpage.

4. Project Workflow
Weather Dataset
       ↓
Data Preprocessing
       ↓
Feature Selection
       ↓
Model Training
       ↓
Prediction
       ↓
Streamlit Dashboard

5. Advantages
Easy to use
Fast prediction
User-friendly interface
Low cost implementation
Educational purpose

6. Limitations
Accuracy depends on dataset quality
Cannot predict extreme weather accurately
Requires sufficient training data

7. Applications
Weather Forecasting
Agriculture
Disaster Management
Research
Educational Projects

## Run Project

streamlit run app.py

## Author
Piyush Kumar
MCA 2177- Galgotias University
