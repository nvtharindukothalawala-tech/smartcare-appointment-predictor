# SmartCare Appointment No-Show Prediction

## 1. Project Overview

SmartCare is a Machine Learning project that predicts whether a patient will attend a scheduled healthcare appointment or miss it.

The main goal is to use appointment-related information to identify appointments that may have a higher risk of being missed. This could help healthcare providers improve appointment planning and resource management.

## 2. Problem Statement

Missed healthcare appointments can waste medical resources and reduce access to healthcare services.

This project explores how Machine Learning can use historical appointment data to predict the likelihood of a patient not attending an appointment.

## 3. Project Objectives

- Explore and understand the dataset.
- Clean and prepare data for Machine Learning.
- Perform Exploratory Data Analysis (EDA).
- Train and compare multiple classification models.
- Evaluate model performance using suitable metrics.
- Explain model predictions using feature importance.
- Develop a Streamlit application for interactive predictions.

## 4. Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Model Saving:** Joblib
- **Web Application:** Streamlit
- **Version Control:** Git and GitHub

## 5. Machine Learning Workflow

1. Load and inspect the appointment dataset.
2. Prepare the input features and target variable.
3. Perform exploratory data analysis.
4. Split the data into training and testing sets.
5. Train Logistic Regression, Random Forest, and Gradient Boosting models.
6. Compare models using cross-validation and test-set metrics.
7. Select a model based on the training cross-validation ROC-AUC score.
8. Save the selected model.
9. Analyze feature contributions.
10. Use the trained model in a Streamlit application.

## 6. Models Evaluated

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

The models are compared using accuracy, precision, recall, F1-score, and ROC-AUC.

The current training script selects the final model using the mean ROC-AUC from five-fold cross-validation on the training data.

## 7. Project Results

The current best model selected by the training cross-validation criterion is Logistic Regression.

| Metric | Logistic Regression |
|---|---:|
| Test Accuracy | 61.5% |
| Test Precision | 61.46% |
| Test Recall | 59.60% |
| Test F1-score | 60.51% |
| Test ROC-AUC | 0.6355 |

These results show that the model has limited predictive performance. Further feature engineering, data quality improvements, and validation on additional data may help improve it.

## 8. Explainability

The project examines Logistic Regression coefficients to understand which input features are associated with higher or lower predicted no-show risk.

These relationships are associations learned from the dataset; they do not establish that a feature causes a patient to miss an appointment.

## 9. Streamlit Application

The application allows users to enter appointment-related information and obtain a model prediction.

The application displays the predicted appointment outcome and the estimated probability of a no-show.

## 10. Project Structure

```text
smartcare-appointment-predictor/
├── data/
│   ├── smartcare_ai_dataset_1000.csv
│   ├── features.csv
│   └── target.csv
├── models/
│   └── best_model.joblib
├── reports/
│   ├── figures/
│   ├── feature_importance.csv
│   └── model_comparison.csv
├── src/
│   ├── app.py
│   ├── eda.py
│   ├── explain_model.py
│   ├── inspect_data.py
│   ├── prepare_data.py
│   └── train_models.py
├── .gitignore
└── README.md
```

## 11. How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/nvtharindukothalawala-tech/smartcare-appointment-predictor.git
cd smartcare-appointment-predictor
```

### Step 2: Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install the required packages

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit application

```bash
streamlit run src/app.py
```

The application should open in your web browser.

## 12. Future Improvements

- Collect and evaluate on more representative appointment data.
- Improve predictive performance through feature engineering and hyperparameter tuning.
- Evaluate class imbalance and probability calibration.
- Add explainable predictions for individual appointments.
- Develop a dashboard for healthcare administrators.
- Deploy the application to a suitable hosting platform.
- Evaluate fairness, privacy, and reliability before any real-world clinical or operational use.

## 13. Disclaimer

This project is intended for educational and research purposes. Predictions should not be used as the sole basis for decisions about patients or access to healthcare services.

## 14. Author

**Tharindu Kothalawala**

B.Sc. (Hons) in Data Science  
Sri Lanka Technological Campus (SLTC)
