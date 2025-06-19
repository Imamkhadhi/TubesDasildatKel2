import streamlit as st
import joblib
import numpy as np

# Load semua model regresi
model_dt = joblib.load("modelJb_DecisionTree_regresi.joblib")
model_knn = joblib.load("modelJb_KNN_regresi.joblib")
model_svm = joblib.load("modelJb_SVM_regresi.joblib")
model_nn = joblib.load("modelJb_NN_regresi.joblib")

def predict_and_show(model_name, model, input_data):
    prediction = model.predict(input_data)
    st.subheader(f"{model_name} Prediction: $ {prediction[0]:,.2f}")

def show_single_regression():
    st.title("🔍 Single Prediction regresi")

    st.markdown("### Input Fitur")
    country = st.number_input("Country (code)", min_value=0)
    page1 = st.number_input("Page 1 (Main Category)", min_value=0)
    page2 = st.number_input("Page 2 (Clothing Model)", min_value=0)
    colour = st.number_input("Colour (code)", min_value=0)
    location = st.number_input("Location (code)", min_value=0)
    photo = st.number_input("Model Photography (0/1)", min_value=0, max_value=1)

    # Checkbox model
    use_knn = st.checkbox("Use KNN")
    use_svm = st.checkbox("Use SVM")
    use_nn = st.checkbox("Use Neural Network")
    use_dt = st.checkbox("Use Decision Tree")

    if st.button("Prediksi Harga"):
        input_data = np.array([[country, page1, page2, colour, location, photo]])

        if use_knn:
            predict_and_show("K-Nearest Neighbors", model_knn, input_data)
        if use_svm:
            predict_and_show("Support Vector Machine", model_svm, input_data)
        if use_nn:
            predict_and_show("Neural Network", model_nn, input_data)
        if use_dt:
            predict_and_show("Decision Tree", model_dt, input_data)

        if not any([use_knn, use_svm, use_nn, use_dt]):  # tambah use_svm jika diaktifkan
            st.warning("Silakan pilih minimal satu model untuk prediksi.")
