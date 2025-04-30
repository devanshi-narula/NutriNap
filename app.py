from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import csv
from datetime import datetime

CSV_FILE = 'user_data.csv'

def save_user_data(row):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp","Age","Gender", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "BMI Category",
                "Heart Rate", "Daily Steps", "Dietary Habits", "Systolic BP", "Diastolic BP",
                "Predicted Health", "Predicted Stress"
            ])
        writer.writerow(row)
        
app = Flask(__name__)

# Load models
health_model = joblib.load('health_model.pkl')
stress_model = joblib.load('stress_model.pkl')

features = ["Age","Gender","Sleep Duration", "Quality of Sleep", "Physical Activity Level", "BMI Category", "Heart Rate",
            "Daily Steps", "Dietary Habits", "Systolic Blood Pressure", "Diastolic Blood Pressure"]

bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
diet_map = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}
gender_map = {"Male": 0, "Female": 1}

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    suggestions = []

    if request.method == 'POST':
        # Collect input data from form
        age = int(request.form['age'])
        gender = request.form['gender']
        sleep_duration = float(request.form['sleep_duration'])
        quality_of_sleep = int(request.form['quality_of_sleep'])
        physical_activity = float(request.form['physical_activity'])
        bmi_category = request.form['bmi_category']
        heart_rate = int(request.form['heart_rate'])
        daily_steps = int(request.form['daily_steps'])
        dietary_habits = request.form['dietary_habits']
        systolic_bp = int(request.form['systolic_bp'])
        diastolic_bp = int(request.form['diastolic_bp'])

        user_data = pd.DataFrame([[age,gender_map[gender],sleep_duration, quality_of_sleep, physical_activity, bmi_map[bmi_category],heart_rate, daily_steps, diet_map[dietary_habits], systolic_bp, diastolic_bp]],columns=features)

        predicted_health = health_model.predict(user_data)[0]
        predicted_stress = stress_model.predict(user_data)[0]

        # Generate suggestions
        if predicted_health < 5:
            suggestions.append("Increase physical activity and improve diet to boost your health index.")
        elif predicted_health < 8:
            suggestions.append("Maintain a balanced routine to sustain your current health level.")
        else:
            suggestions.append("Great job! Keep up your healthy habits.")

        if predicted_stress > 6:
            suggestions.append("Consider relaxation techniques like meditation or better sleep hygiene to reduce stress.")
        elif predicted_stress > 3:
            suggestions.append("Try managing your workload and getting adequate rest to lower stress levels.")
        else:
            suggestions.append("Your stress levels are in a good range. Keep managing it well.")

        result = {
            "predicted_health": round(predicted_health, 2),
            "predicted_stress": round(predicted_stress, 2),
            "suggestions": suggestions
        }

        # Save to CSV
        with open('user_data.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                age,
                gender,
                sleep_duration,
                quality_of_sleep,
                physical_activity,
                bmi_category,
                heart_rate,
                daily_steps,
                dietary_habits,
                systolic_bp,
                diastolic_bp,
                round(predicted_health, 2),
                round(predicted_stress, 2)
            ])

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
