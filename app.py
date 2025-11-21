import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models & preprocessing
preprocess = joblib.load("preprocess.pkl")
model_logreg = joblib.load("model_logreg.pkl")
model_rf = joblib.load("model_rf.pkl")
model_voting = joblib.load("model_voting.pkl")

# Judul
st.title("❤️ Heart Disease Prediction App")
st.write("Aplikasi cerdas untuk memprediksi penyakit jantung menggunakan dua model Machine Learning dan Voting Ensemble.")

# Sidebar – pilih model
st.sidebar.header("🔍 Pilih Model Prediksi")
model_choice = st.sidebar.selectbox(
    "Pilih model:",
    ("Logistic Regression", "Random Forest", "Voting Classifier")
)

# Input form
st.header("📥 Masukkan Data Pasien")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
chol = st.number_input("Cholesterol", 50, 700, 200)
thalach = st.number_input("Max Heart Rate", 50, 250, 150)
restingbp = st.number_input("Resting BP", 80, 200, 120)
fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
exerciseangina = st.selectbox("Exercise Angina", ["Y", "N"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
oldpeak = st.number_input("Oldpeak", -2.0, 6.0, 1.0)
slope = st.selectbox("Slope", ["Up", "Flat", "Down"])

# Buat dataframe input user
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "ChestPainType": [cp],
    "Cholesterol": [chol],
    "MaxHR": [thalach],
    "RestingBP": [restingbp],
    "FastingBS": [fastingbs],
    "ExerciseAngina": [exerciseangina],
    "RestingECG": [restecg],
    "Oldpeak": [oldpeak],
    "ST_Slope": [slope]
})

# Preprocessing
processed_input = preprocess.transform(input_data)

# Prediksi
if model_choice == "Logistic Regression":
    model = model_logreg
elif model_choice == "Random Forest":
    model = model_rf
else:
    model = model_voting

prediction = model.predict(processed_input)[0]

# Output hasil
st.header("📊 Hasil Prediksi")

if prediction == 1:
    st.error("⚠️ Pasien Berisiko Penyakit Jantung")
else:
    st.success("✅ Pasien Tidak Berisiko Penyakit Jantung")

# Tampilkan akurasi (dari training)
st.sidebar.subheader("📈 Akurasi Model")
st.sidebar.write("Logistic Regression: **±91%**")
st.sidebar.write("Random Forest: **±94%**")
st.sidebar.write("Voting Classifier: **±96%** (Model Terbaik)")

