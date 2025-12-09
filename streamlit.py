import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

st.title("ML Project Dashboard")

st.write("Upload your dataset, train a model, and generate predictions.")

# ---------------------------
# 1. DATA UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target_col = st.selectbox("Select target column", df.columns)

    if target_col:
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # ---------------------------
        # 2. TRAIN TEST SPLIT
        # ---------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---------------------------
        # 3. TRAIN MODEL
        # ---------------------------
        if st.button("Train Model"):
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Predictions
            preds = model.predict(X_test)

            # ---------------------------
            # 4. METRICS
            # ---------------------------
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            st.subheader("Model Performance")
            st.write("Accuracy:", acc)
            st.write("F1 Score:", f1)

            # ---------------------------
            # 5. DOWNLOAD PREDICTIONS
            # ---------------------------
            pred_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": preds
            })

            st.subheader("Prediction Results")
            st.dataframe(pred_df)

            csv_download = pred_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Predictions CSV",
                data=csv_download,
                file_name="predictions.csv",
                mime="text/csv"
            )
