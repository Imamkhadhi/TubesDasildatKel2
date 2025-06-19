import streamlit as st
from SinglePrediction import show_single
from SinglePredictionRegresi import show_single_regression
from BatchUpload import show_batch
from BatchUploadRegresi import show_batch_regression

st.title("👕 Dataset E-Clothes Dashboards")

menu = st.sidebar.radio("Choose a mode:", ["Single Prediction (klasifikasi)", "Single Prediction (regresi)", "Batch Upload (klasifikasi)", "Batch Upload (Regresi)"])
#menu = st.radio("Choose a mode:", ["Single Prediction", "Batch Upload"])

if menu == "Single Prediction (klasifikasi)":
    show_single()
elif menu == "Single Prediction (regresi)":
    show_single_regression()
elif menu == "Batch Upload (klasifikasi)":
    show_batch()
elif menu == "Batch Upload (Regresi)":
    show_batch_regression()
