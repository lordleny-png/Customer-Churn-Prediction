# 📊 Customer Churn Prediction

## Project Overview

Customer churn is a major challenge for telecommunications companies because losing customers can reduce revenue and increase customer acquisition costs.

This project uses machine learning to predict customer churn based on customer demographics, subscribed services, contract information, tenure, and billing details. The objective is to help businesses identify customers at risk of leaving and develop targeted retention strategies.

An interactive dashboard built with **Streamlit** allows users to enter customer information and obtain a churn prediction.

## 🎯 Objectives

- Analyze customer characteristics and identify patterns associated with churn.
- Explore relationships between churn, contract type, customer tenure, and monthly charges.
- Develop and evaluate machine learning classification models.
- Adjust the classification threshold to improve the detection of customers likely to churn.
- Build an interactive application to support data-driven customer retention decisions.

## 📂 Dataset

The project uses the Telco Customer Churn dataset, originally published on Kaggle.

**Dataset source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)

The original dataset contains 7,043 customer records and 21 columns. After converting `TotalCharges` to numeric values and removing 11 records with missing values in that field, the analysis dataset contains **7,032 records and 20 columns**, including the target variable.

The dataset includes customer demographics, services, contract information, payment methods, tenure, and billing amounts.

**Target variable:** `Churn`
- `Yes`: The customer churned.
- `No`: The customer did not churn.

## 🛠️ Technologies Used

- **Python** — data analysis and model development
- **Pandas & NumPy** — data manipulation and numerical computing
- **Matplotlib** — data visualization
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Streamlit** — interactive web application
- **Git & GitHub** — version control and project documentation

## 🔍 Exploratory Data Analysis

The exploratory analysis revealed several patterns:

- **Contract type:** Month-to-month customers had a churn rate of approximately 42.71%, compared with 11.28% for one-year contracts and 2.85% for two-year contracts.
- **Customer tenure:** Customers who churned had an average tenure of approximately 18 months, compared with 38 months for customers who stayed.
- **Monthly charges:** Customers who churned paid an average of approximately $74.44 per month, compared with $61.31 for customers who stayed.

These findings highlight customer groups that may benefit from targeted retention initiatives. They describe associations in the dataset and do not establish causation.

## 🤖 Machine Learning Methodology

### 1. Data Preparation
- Removed the customer identifier from the predictive features.
- Converted `TotalCharges` to a numeric format.
- Removed records with missing total charges.
- Separated the predictors from the target variable.
- Split the data into training and testing sets using an 80/20 split with stratification.
- Standardized numerical features and one-hot encoded categorical features.

### 2. Model Development

Two classification algorithms were evaluated:

- **Logistic Regression**
- **Random Forest Classifier**

### 3. Model Evaluation

The models were assessed using accuracy, precision, recall, F1-score, and ROC-AUC.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.38% | 64.85% | 57.22% | 60.80% | 83.59% |
| Random Forest | 77.54% | 57.04% | 62.83% | 59.80% | 82.50% |

Based on these results, Logistic Regression achieved higher accuracy, precision, F1-score, and ROC-AUC, while Random Forest achieved higher recall.

### 4. Classification Threshold Optimization

The default classification threshold of 0.50 was compared with a lower threshold of 0.40 for Logistic Regression.

At a threshold of 0.40, the model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 78.39% |
| Precision | 57.92% |
| Recall | 68.45% |
| F1-score | 62.75% |

The lower threshold increased recall, helping the model identify more customers who churned, at the cost of lower precision and accuracy.

The threshold of 0.40 was selected as a candidate for customer retention use cases, where missing a potential churner may be costly.

## 📈 Interactive Dashboard

### Dashboard Preview

![Customer Churn Prediction Dashboard](churn_dashboard.png)          



The Streamlit application allows users to:

- Enter customer demographic and service information.
- Specify contract, tenure, and billing details.
- Estimate the probability of customer churn.
- Classify a customer as likely or unlikely to churn using the selected threshold.

**Live application:** [Open the Customer Churn Prediction Dashboard](https://customer-churn-prediction-byx3hofdya5lhgkkpsfcvz.streamlit.app/)

Replace the placeholder above with your actual Streamlit Cloud URL.

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/lordleny-png/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

Ensure the model JSON file is present in the repository and that its filename matches the one referenced in `app.py`.

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
Customer-Churn-Prediction/
├── app.py
├── churn_model_.json
├── requirements.txt
└── README.md
```

The model JSON filename shown above reflects the current application setup. Update the filename in this structure if you rename the file in GitHub.

## 💼 Business Value

Customer churn predictions can help telecommunications businesses:

- Prioritize customers who may need additional support.
- Design targeted customer retention campaigns.
- Identify customer segments associated with higher churn.
- Support customer relationship management and marketing decisions.
- Allocate retention resources more efficiently.

The predictions should support business decisions rather than replace human judgment. The model should be monitored and validated before being used in operational decisions.

## ⚠️ Limitations

- The model is trained on a specific telecommunications dataset and may not generalize to other companies without additional validation.
- Observational patterns do not establish the causes of customer churn.
- Prediction quality depends on data quality, feature consistency, and the classification threshold.
- The threshold and model should be reassessed as customer behavior changes.

## 👤 Author

**Leny Napala**

Business Analytics | Data Science | Machine Learning

Interested in applying statistical analysis and machine learning to business
