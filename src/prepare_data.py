
import pandas as pd

# 1. Load the original dataset
file_path = "data/smartcare_ai_dataset_1000.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# 2. Remove columns that reveal the outcome or are unsuitable
# for prediction before the appointment
columns_to_drop = [
    "record_id",
    "patient_id",
    "appointment_status",
    "admitted",
    "room_type",
    "length_of_stay_days",
    "previous_admissions",
    "consultation_fee_lkr",
    "room_charge_lkr",
    "lab_charge_lkr",
    "medicine_charge_lkr",
    "total_bill_lkr",
    "payment_status",
    "payment_method",
    "readmitted_30_days",
    "disease_risk_level",
]

# Drop only columns that exist in the dataset
df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)

# 3. Convert appointment_date into useful date features
df["appointment_date"] = pd.to_datetime(
    df["appointment_date"],
    errors="coerce"
)

df["appointment_month"] = df["appointment_date"].dt.month
df["appointment_day_of_week"] = (
    df["appointment_date"].dt.day_of_week
)

# Remove the original date column
df = df.drop(columns=["appointment_date"])

# 4. Separate features (X) and target (y)
X = df.drop(columns=["no_show"])
y = df["no_show"]

# 5. Display the results
print("\nPrepared feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nRemaining features:")
print(X.columns.tolist())

print("\nMissing values in features:")
print(X.isnull().sum()[X.isnull().sum() > 0])

print("\nTarget distribution:")
print(y.value_counts())

# 6. Save the prepared data
X.to_csv("data/features.csv", index=False)
y.to_csv("data/target.csv", index=False)

print("\nPrepared files saved successfully!")
