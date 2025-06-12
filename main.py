import streamlit as st
from SinglePrediction import show_single
from SinglePredictionRegresi import show_single_regression
from BatchUpload import show_batch
from BatchUploadRegresi import show_batch_regression

st.title("👕 Dataset E-Clothes Dashboard")

menu = st.sidebar.radio("Choose a mode:", ["Single Prediction", "Regresi (Single)", "Batch Upload", "Batch Upload (Regresi)"])
#menu = st.radio("Choose a mode:", ["Single Prediction", "Batch Upload"])

if menu == "Single Prediction":
    show_single()
elif menu == "Regresi (Single)":
    show_single_regression()
elif menu == "Batch Upload":
    show_batch()
elif menu == "Batch Upload (Regresi)":
    show_batch_regression()