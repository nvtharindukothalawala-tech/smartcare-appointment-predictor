
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the original dataset
df = pd.read_csv("data/smartcare_ai_dataset_1000.csv")

# Create folder to save charts
os.makedirs("reports/figures", exist_ok=True)

sns.set_theme(style="whitegrid")

# Chart 1: Appointment outcomes
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="no_show")
plt.title("Appointment Outcomes")
plt.xlabel("0 = Attended, 1 = No-show")
plt.ylabel("Number of Appointments")
plt.tight_layout()
plt.savefig("reports/figures/01_appointment_outcomes.png")
plt.close()

# Chart 2: Waiting days
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="no_show", y="waiting_days")
plt.title("Waiting Days by Appointment Outcome")
plt.xlabel("0 = Attended, 1 = No-show")
plt.ylabel("Waiting Days")
plt.tight_layout()
plt.savefig("reports/figures/02_waiting_days.png")
plt.close()

# Chart 3: Age groups
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 18, 35, 50, 65, 120],
    labels=["0-18", "19-35", "36-50", "51-65", "66+"]
)

age_rates = df.groupby(
    "age_group", observed=False
)["no_show"].mean() * 100

plt.figure(figsize=(7, 5))
age_rates.plot(kind="bar")
plt.title("No-show Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("No-show Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/figures/03_age_group.png")
plt.close()

# Chart 4: Previous missed appointments
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="missed_previous_appointments",
    y="no_show",
    errorbar=None
)
plt.title("No-show Rate by Previous Missed Appointments")
plt.xlabel("Previous Missed Appointments")
plt.ylabel("No-show Rate")
plt.tight_layout()
plt.savefig("reports/figures/04_previous_missed.png")
plt.close()

# Chart 5: Department
department_rates = (
    df.groupby("department")["no_show"]
    .mean()
    .sort_values(ascending=False) * 100
)

plt.figure(figsize=(9, 5))
department_rates.plot(kind="bar")
plt.title("No-show Rate by Department")
plt.xlabel("Department")
plt.ylabel("No-show Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("reports/figures/05_department.png")
plt.close()

print("EDA completed successfully!")
print("Charts saved in reports/figures/")

print("\nNo-show rate by department:")
print(department_rates.round(2))

print("\nNo-show rate by age group:")
print(age_rates.round(2))
