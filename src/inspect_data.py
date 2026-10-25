
import pandas as pd

# Load the dataset
file_path = "data/smartcare_ai_dataset_1000.csv"
df = pd.read_csv(file_path)

print("=" * 50)
print("SMARTCARE DATASET INSPECTION")
print("=" * 50)

# 1. Dataset size
print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Missing values
print("\n2. MISSING VALUES")
missing = df.isnull().sum()
print(missing[missing > 0])

# 3. Target distribution
print("\n3. NO-SHOW DISTRIBUTION")
print(df["no_show"].value_counts())
print("\nTarget percentages:")
print((df["no_show"].value_counts(normalize=True) * 100).round(2))

# 4. Duplicate records
print("\n4. DUPLICATE ROWS")
print(df.duplicated().sum())

# 5. Data types
print("\n5. COLUMN DATA TYPES")
print(df.dtypes)

# 6. Check appointment status against the target
print("\n6. APPOINTMENT STATUS VS NO-SHOW")
if "appointment_status" in df.columns:
    print(pd.crosstab(df["appointment_status"], df["no_show"]))

print("\nInspection completed successfully!")
