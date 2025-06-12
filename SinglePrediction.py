import streamlit as st
import joblib
import numpy as np

# Load models once
model_dt = joblib.load("modelJb_DecisionTree_klasifikasi.joblib")
model_knn = joblib.load("modelJb_KNN_klasifikasi.joblib")
model_svm = joblib.load("modelJb_SVM_klasifikasi.joblib")
model_nn = joblib.load("modelJb_NN_klasifikasi.joblib")

# Label prediksi
label_map = {
    1: "Price Class 1",
    2: "Price Class 2"
}

def predict_and_show(model_name, model, input_data):
    pred = model.predict(input_data)
    label = label_map.get(pred[0], "Unknown")
    st.subheader(f"{model_name} Prediction: {pred[0]} → {label}")

# ✅ Ganti nama dari `main()` ke `show_single()`
def show_single():
    st.header("🔍 Single Prediction")

    # Input user untuk 7 fitur
    country = st.number_input("Country (code)", min_value=0)
    page1 = st.number_input("Page 1 (Main Category)", min_value=0)
    page2 = st.number_input("Page 2 (Clothing Model) — Encoded", min_value=0)
    colour = st.number_input("Colour (code)", min_value=0)
    location = st.number_input("Location (code)", min_value=0)
    photo = st.number_input("Model Photography (0/1)", min_value=0, max_value=1)

    # Model checkboxes
    use_knn = st.checkbox("Use KNN")
    use_svm = st.checkbox("Use SVM")
    use_nn = st.checkbox("Use Neural Network")
    use_dt = st.checkbox("Use Decision Tree")

    if st.button("Predict"):
        input_data = np.array([[country, page1, page2, colour, location, photo]])

        if use_knn:
            predict_and_show("K-Nearest Neighbors", model_knn, input_data)
        if use_svm:
            predict_and_show("Support Vector Machine", model_svm, input_data)
        if use_nn:
            predict_and_show("Neural Network", model_nn, input_data)
        if use_dt:
            predict_and_show("Decision Tree", model_dt, input_data)

        if not any([use_knn, use_svm, use_nn, use_dt]):
            st.warning("Please select at least one model.")
