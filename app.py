import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# ======================================
# LOAD DATASET (langsung dari Github kamu)
# ======================================
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

df = load_data()

st.title("❤️ Heart Failure Prediction App (Tanpa .pkl, Anti Error)")
st.write("Model dilatih langsung di Streamlit agar tidak terjadi error versi Python.")

# ======================================
# PREPROCESSING + TRAINING LANGSUNG
# ======================================
categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
numeric = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(), categorical)
])

logreg = LogisticRegression(max_iter=200)
rf = RandomForestClassifier(n_estimators=200)

voting = VotingClassifier(
    estimators=[("lr", logreg), ("rf", rf)],
    voting="soft"
)

model = Pipeline([
    ("prep", preprocess),
    ("model", voting)
])

model.fit(X, y)

# ======================================
# INPUT FORM
# ======================================

st.subheader("Masukkan Data Pasien")

age = st.number_input("Age", 20, 90, 45)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting BP", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
rest_ecg = st.selectbox("Rest ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max HR", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "ChestPainType": [chest_pain],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [fasting_bs],
    "RestingECG": [rest_ecg],
    "MaxHR": [max_hr],
    "ExerciseAngina": [exercise_angina],
    "Oldpeak": [oldpeak],
    "ST_Slope": [st_slope]
})

# ======================================
# PREDIKSI
# ======================================

if st.button("Prediksi"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    st.subheader("Hasil Prediksi:")

    if pred == 1:
        st.error(f"🚨 Pasien Berisiko Tinggi (Probabilitas: {prob:.2f}%)")
    else:
        st.success(f"✅ Pasien Tidak Berisiko (Probabilitas: {prob:.2f}%)")


