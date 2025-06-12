import streamlit as st
import pandas as pd
import joblib

def show_batch_regression():
    st.header("📦 Batch Upload - Price Prediction (Regresi)")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### 📄 Uploaded Data", df.head())

        # Validasi kolom
        required_columns = ['country', 'page 1 (main category)', 'page 2 (clothing model)',
                            'colour', 'location', 'model photography']
        if not all(col in df.columns for col in required_columns):
            st.error(f"❌ Kolom tidak sesuai. Pastikan CSV memiliki kolom berikut:\n{required_columns}")
            return

        X = df[required_columns]

        # Pilih model
        st.subheader("🤖 Pilih Model Regresi")
        use_knn = st.checkbox("Use K-Nearest Neighbors (Regresi)")
        use_dt  = st.checkbox("Use Decision Tree (Regresi)")
        use_nn  = st.checkbox("Use Neural Network (Regresi)")
        use_svm = st.checkbox("Use Support Vector Machine (Regresi)")

        if st.button("🔍 Prediksi Harga"):
            if not any([use_knn, use_dt, use_nn]):
                st.warning("❗ Silakan pilih setidaknya satu model.")
                return

            if use_knn:
                model = joblib.load("modelJb_KNN_regresi.joblib")
                preds = model.predict(X)
                df['KNN Price'] = preds
                st.markdown("### 💰 Hasil Prediksi: KNN Regresi")
                st.dataframe(df[['KNN Price']])

            if use_dt:
                model = joblib.load("modelJb_DecisionTree_rergresi.joblib")
                preds = model.predict(X)
                df['DT Price'] = preds
                st.markdown("### 💰 Hasil Prediksi: Decision Tree Regresi")
                st.dataframe(df[['DT Price']])

            if use_nn:
                model = joblib.load("modelJb_NN_regresi.joblib")
                preds = model.predict(X)
                df['NN Price'] = preds
                st.markdown("### 💰 Hasil Prediksi: Neural Network Regresi")
                st.dataframe(df[['NN Price']])
            
            if use_svm:
                model = joblib.load("modelJb_SVM_regresi.joblib")
                preds = model.predict(X)
                df['SVM Price'] = preds
                st.markdown("### 💰 Hasil Prediksi: Support Vector Machine Regresi")
                st.dataframe(df[['SVM Price']])

            # Optional download
            # st.download_button("💾 Download Prediksi", df.to_csv(index=False), "prediksi_harga.csv", "text/csv")
