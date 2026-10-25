
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Load the trained model
model = joblib.load("models/best_model.joblib")

# Load the original dataset
df = pd.read_csv("data/smartcare_ai_dataset_1000.csv")

# Use the same features used during model training
df["appointment_date"] = pd.to_datetime(df["appointment_date"])
df["appointment_month"] = df["appointment_date"].dt.month
df["appointment_day_of_week"] = df["appointment_date"].dt.day_of_week

features = [
    "age",
    "gender",
    "blood_group",
    "department",
    "waiting_days",
    "previous_appointments",
    "missed_previous_appointments",
    "appointment_month",
    "appointment_day_of_week",
]

X = df[features]

# Get the preprocessing and Logistic Regression model
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

# Transform the data
X_transformed = preprocessor.transform(X)

# Get the feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get the Logistic Regression coefficients
importance = classifier.coef_[0]

# Create a feature importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": importance,
    "Absolute_Importance": abs(importance),
})

importance_df = importance_df.sort_values(
    by="Absolute_Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance_df.head(10).to_string(index=False))

# Save the results
os.makedirs("reports/figures", exist_ok=True)
importance_df.to_csv("reports/feature_importance.csv", index=False)

# Plot the top 10 features
top_features = importance_df.head(10).sort_values(
    by="Absolute_Importance"
)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)
plt.title("Logistic Regression Feature Importance")
plt.xlabel("Coefficient (positive = higher no-show odds)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(
    "reports/figures/feature_importance.png",
    dpi=150
)
plt.close()

print("\nSaved: reports/feature_importance.csv")
print("Saved: reports/figures/feature_importance.png")
