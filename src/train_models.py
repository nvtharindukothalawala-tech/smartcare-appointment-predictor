
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/smartcare_ai_dataset_1000.csv")

print("=" * 60)
print("SMARTCARE APPOINTMENT PREDICTOR")
print("MACHINE LEARNING MODEL TRAINING")
print("=" * 60)

# Remove records without a target value
df = df.dropna(subset=["no_show"]).copy()

# Create date-based features
df["appointment_date"] = pd.to_datetime(
    df["appointment_date"],
    errors="coerce"
)

df["appointment_month"] = df["appointment_date"].dt.month
df["appointment_day_of_week"] = (
    df["appointment_date"].dt.day_of_week
)

# --------------------------------------------------
# 2. SELECT PRE-APPOINTMENT FEATURES
# --------------------------------------------------

# Use only features intended to be available before
# the appointment. Exclude appointment outcomes,
# clinical measurements, billing and later events.

feature_columns = [
    "age",
    "gender",
    "blood_group",
    "department",
    "waiting_days",
    "previous_appointments",
    "missed_previous_appointments",
    "appointment_month",
    "appointment_day_of_week"
]

X = df[feature_columns].copy()
y = df["no_show"].astype(int)

print("\nDataset rows:", len(df))
print("Number of features:", len(feature_columns))
print("Target distribution:")
print(y.value_counts().sort_index())

# --------------------------------------------------
# 3. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

numeric_features = [
    "age",
    "waiting_days",
    "previous_appointments",
    "missed_previous_appointments",
    "appointment_month",
    "appointment_day_of_week"
]

categorical_features = [
    "gender",
    "blood_group",
    "department"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 5. DEFINE THREE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

results = []
trained_models = {}
cv_scores = {}

# --------------------------------------------------
# 6. TRAIN AND EVALUATE
# --------------------------------------------------

for name, model in models.items():

    print("\n" + "-" * 50)
    print("MODEL:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Cross-validation uses training data only
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    )

    cv_scores[name] = scores.mean()

    print("Mean 5-fold CV ROC-AUC:", round(scores.mean(), 4))

    # Fit using the complete training split
    pipeline.fit(X_train, y_train)

    # Predict on the untouched test split
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    print("Test Accuracy:", round(accuracy, 4))
    print("Test Precision:", round(precision, 4))
    print("Test Recall:", round(recall, 4))
    print("Test F1-score:", round(f1, 4))
    print("Test ROC-AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Attended", "No-show"],
        zero_division=0
    ))

    results.append({
        "Model": name,
        "CV ROC-AUC": scores.mean(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = pipeline

    # Save confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Attended", "No-show"],
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        f"reports/figures/confusion_matrix_{filename}.png"
    )
    plt.close()

    # Save ROC curve
    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name=name
    )

    plt.title(f"ROC Curve - {name}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/roc_curve_{filename}.png")
    plt.close()

# --------------------------------------------------
# 7. COMPARE MODELS
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "CV ROC-AUC",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# Select the model using training cross-validation,
# not by choosing whichever test score looks best.
best_name = max(cv_scores, key=cv_scores.get)
best_model = trained_models[best_name]

joblib.dump(best_model, "models/best_model.joblib")

print("\nSelected model:", best_name)
print("Selection basis: highest mean training CV ROC-AUC")
print("Saved model: models/best_model.joblib")
print("Saved metrics: reports/model_comparison.csv")
print("Saved charts: reports/figures/")
print("\nMODEL TRAINING COMPLETED!")
