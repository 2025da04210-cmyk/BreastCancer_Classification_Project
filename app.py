import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt


# Title
st.title("Breast Cancer Classification")


# Model folder
model_folder = "model"


# Available models
models = {
    "Logistic Regression": "Logistic_Model.pkl",
    "Random Forest": "Random_Forest_Model.pkl",
    "SVM": "SVM_Model.pkl",
    "Decision Tree": "Decision_Tree_Model.pkl",
    "Naive Bayes ":"Gaussian_Naive_Bayes_Model.pkl"
}


# Model selection dropdown
selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)


# CSV upload option
uploaded_file = st.file_uploader(
    "Upload Test Dataset CSV",
    type="csv"
)


if uploaded_file is not None:

    # Read test data
    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.write(test_data.head())


    # Target column
    target_column = "diagnosis"


    # Split features and target
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]


    # Load selected model
    model_path = os.path.join(
        model_folder,
        models[selected_model]
    )

    model = joblib.load(model_path)


    # Prediction button
    if st.button("Predict"):

        prediction = model.predict(X_test)


        # Evaluation Metrics
        st.subheader("Evaluation Metrics")


        accuracy = accuracy_score(
            y_test,
            prediction
        )

        precision = precision_score(
            y_test,
            prediction,
            average="weighted"
        )

        recall = recall_score(
            y_test,
            prediction,
            average="weighted"
        )

        f1 = f1_score(
            y_test,
            prediction,
            average="weighted"
        )


        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1 Score:", f1)



        # Classification Report
        st.subheader("Classification Report")


        report = classification_report(
            y_test,
            prediction
        )

        st.text(report)



        # Confusion Matrix
        st.subheader("Confusion Matrix")


        cm = confusion_matrix(
            y_test,
            prediction
        )


        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            ax=ax
        )


        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")


        st.pyplot(fig)


else:

    st.info("Please upload test_data.csv file to continue")
