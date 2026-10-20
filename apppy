
import json
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model parameters
@st.cache_data
def load_model():
    with open("churn_model.json", "r") as file:
        return json.load(file)

model = load_model()

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information to estimate the probability "
    "of churn using a Logistic Regression model."
)

# Customer information form
with st.form("customer_form"):
    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=100, value=12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )

    with col2:
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    col3, col4 = st.columns(2)

    with col3:
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col4:
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, value=70.0
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=840.0
        )

    submitted = st.form_submit_button("Predict Churn")


# Make a prediction using the saved JSON parameters
if submitted:
    try:
        row = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        customer = pd.DataFrame([row])

        # Transform numeric features
        numeric = customer[model["numerical_features"]].astype(float)
        numeric = numeric.fillna(
            pd.Series(model["numerical_imputer_medians"])
        )

        numeric = (
            numeric - np.array(model["scaler_means"])
        ) / np.array(model["scaler_scales"])

        # Transform categorical features using saved categories
        categorical = customer[model["categorical_features"]].copy()

        for i, column in enumerate(model["categorical_features"]):
            categories = model["categorical_categories"][i]
            value = categorical.iloc[0][column]

            if value not in categories:
                raise ValueError(
                    f"Unsupported value for {column}: {value}"
                )

        encoded = []
        for i, column in enumerate(model["categorical_features"]):
            categories = model["categorical_categories"][i]
            value = categorical.iloc[0][column]
            encoded.extend([1.0 if value == category else 0.0
                            for category in categories])

        # Combine features in the same order used during training
        numeric_values = numeric.to_numpy().ravel()
        features = np.concatenate([numeric_values, encoded])

        coefficients = np.array(model["coefficients"])
        intercept = model["intercept"]

        if len(features) != len(coefficients):
            raise ValueError(
                "Feature count does not match the trained model."
            )

        # Logistic Regression probability
        score = np.dot(features, coefficients) + intercept
        probability = 1 / (1 + np.exp(-np.clip(score, -500, 500)))

        threshold = model["threshold"]
        prediction = "Yes" if probability >= threshold else "No"

        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{probability:.1%}")

        if prediction == "Yes":
            st.warning(
                "⚠️ This customer is classified as likely to churn."
            )
        else:
            st.success(
                "✅ This customer is classified as unlikely to churn."
            )

        st.caption(f"Classification threshold: {threshold:.0%}")

    except Exception as error:
        st.error(f"Prediction error: {error}")
