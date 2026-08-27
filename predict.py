import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("placement_model.pkl")
scaler = joblib.load("scaler.pkl")


def predict_result(cgpa, attendance, coding_score, projects, internship):
    """
    Predict whether a student will be placed.

    Returns:
        "PLACED" or "NOT PLACED"
    """

    # Create input DataFrame
    data = pd.DataFrame([{
        "CGPA": cgpa,
        "Attendance": attendance,
        "CodingScore": coding_score,
        "Projects": projects,
        "Internship": internship
    }])

    # Scale input
    data_scaled = scaler.transform(data)

    # Make prediction
    prediction = model.predict(data_scaled)[0]

    # Convert 0/1 into readable result
    if prediction == 1:
        return "PLACED"
    else:
        return "NOT PLACED"