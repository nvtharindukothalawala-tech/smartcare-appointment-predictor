
import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="SmartCare Appointment Predictor",
    page_icon="🏥",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")

model = load_model()

# Application title
st.title("🏥 SmartCare Appointment Predictor")
st.write(
    "This application predicts whether a patient may miss "
    "a hospital appointment using machine learning."
)

st.info(
    "This is an academic project. Predictions are estimates "
    "and should not be used as the only basis for healthcare decisions."
)

# Appointment details form
st.header("Enter Appointment Details")

with st.form("appointment_form"):

    age = st.number_input(
        "Patient Age",
        min_value=0,
        max_value=120,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    blood_group = st.selectbox(
        "Blood Group",
        ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    )

    department = st.selectbox(
        "Department",
        [
            "Cardiology",
            "General Medicine",
            "Laboratory Services",
            "Neurology",
            "Orthopedics",
            "Pediatrics",
            "Radiology"
        ]
    )

    waiting_days = st.number_input(
        "Waiting Days",
        min_value=0,
        max_value=365,
        value=7
    )

    previous_appointments = st.number_input(
        "Previous Appointments",
        min_value=0,
        max_value=100,
        value=2
    )

    missed_previous_appointments = st.number_input(
        "Previously Missed Appointments",
        min_value=0,
        max_value=100,
        value=0
    )

    appointment_month = st.selectbox(
        "Appointment Month",
        list(range(1, 13)),
        index=0
    )

    appointment_day_of_week = st.selectbox(
        "Appointment Day",
        list(range(7)),
        format_func=lambda day: [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ][day]
    )

    submitted = st.form_submit_button("Predict Appointment Outcome")

# Make prediction
if submitted:

    if missed_previous_appointments > previous_appointments:
        st.error(
            "Previously missed appointments cannot exceed "
            "the total previous appointments."
        )

    else:
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "blood_group": blood_group,
            "department": department,
            "waiting_days": waiting_days,
            "previous_appointments": previous_appointments,
            "missed_previous_appointments": missed_previous_appointments,
            "appointment_month": appointment_month,
            "appointment_day_of_week": appointment_day_of_week
        }])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        no_show_probability = probabilities[
            list(model.classes_).index(1)
        ]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning("Prediction: No-show")
        else:
            st.success("Prediction: Attended")

        st.metric(
            "Estimated No-show Probability",
            f"{no_show_probability:.1%}"
        )

        st.progress(float(no_show_probability))

        st.caption(
            "This probability is a model estimate, not a guarantee. "
            "Model performance is currently limited and should be "
            "improved and validated before real-world use."
        )
