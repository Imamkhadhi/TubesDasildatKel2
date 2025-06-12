import streamlit as st
import pandas as pd
import joblib

# Mapping label output
label_map = {
    1: "Price Class 1",
    2: "Price Class 2"
}

def show_batch():
    st.header("📦 Batch Upload - Price Class Prediction")

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
        st.subheader("🤖 Pilih Model")
        use_knn = st.checkbox("Use K-Nearest Neighbors")
        # use_svm = st.checkbox("Use Support Vector Machine")
        # use_nn  = st.checkbox("Use Neural Network")
        use_dt  = st.checkbox("Use Decision Tree")

        if st.button("🔍 Prediksi"):
            if not any([use_knn, use_dt]):
                st.warning("❗ Silakan pilih setidaknya satu model.")
                return

            if use_knn:
                model = joblib.load("modelJb_KNN_klasifikasi.joblib")
                preds = model.predict(X)
                df['KNN Class'] = preds
                df['KNN Label'] = [label_map.get(p, "Unknown") for p in preds]
                st.markdown("### ✅ Hasil Prediksi: K-Nearest Neighbors")
                st.dataframe(df[['KNN Class', 'KNN Label']])

            if use_dt:
                model = joblib.load("modelJb_DecisionTree_klasifikasi.joblib")
                preds = model.predict(X)
                df['DT Class'] = preds
                df['DT Label'] = [label_map.get(p, "Unknown") for p in preds]
                st.markdown("### ✅ Hasil Prediksi: Decision Tree")
                st.dataframe(df[['DT Class', 'DT Label']])

            # Tambahkan ini kalau ingin simpan hasil
            # st.download_button("💾 Download CSV", df.to_csv(index=False), "predicted_result.csv", "text/csv")
