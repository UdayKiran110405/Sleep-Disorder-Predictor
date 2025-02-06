import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model, encoders, and scaler
model_filename = "best_sleep_disorder_model.pkl"
encoders_filename = "encoders.pkl"
scaler_filename = "scaler.pkl"

# Load the model
with open(model_filename, "rb") as model_file:
    model = pickle.load(model_file)

# Load the encoders
with open(encoders_filename, "rb") as encoders_file:
    encoders = pickle.load(encoders_file)

# Load the scaler
with open(scaler_filename, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Disorder names mapping (you can adjust this as per your model's output classes)
disorder_names = {
    0: "No Sleep Disorder",
    1: "Insomnia",
    2: "Sleep Apnea",
    3: "Restless Leg Syndrome",  # Add more as per your model's classes
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    gender = request.form['Gender']
    age = float(request.form['Age'])
    occupation = request.form['Occupation']
    sleep_duration = float(request.form['Sleep Duration'])
    quality_of_sleep = float(request.form['Quality of Sleep'])
    physical_activity_level = float(request.form['Physical Activity Level'])
    stress_level = float(request.form['Stress Level'])
    bmi_category = request.form['BMI Category']
    blood_pressure = request.form['Blood Pressure']
    heart_rate = float(request.form['Heart Rate'])
    daily_steps = float(request.form['Daily Steps'])

    # Apply the label encoders (using the loaded encoders)
    gender_encoded = encoders['Gender'].transform([gender])[0]
    occupation_encoded = encoders['Occupation'].transform([occupation])[0]
    bmi_category_encoded = encoders['BMI Category'].transform([bmi_category])[0]
    blood_pressure_encoded = encoders['Blood Pressure'].transform([blood_pressure])[0]

    # Prepare the feature array (only include numerical columns for scaling)
    feature_array = np.array([[gender_encoded, age, occupation_encoded, sleep_duration, quality_of_sleep,
                               physical_activity_level, stress_level, bmi_category_encoded, blood_pressure_encoded,
                               heart_rate, daily_steps]])

    # Only scale numerical columns (since these are the columns the scaler was fit on)
    numerical_columns = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
    feature_array_for_scaling = feature_array[:, [1, 3, 4, 5, 6, 9, 10]]  # Only numerical columns for scaling

    # Scale the features (using the loaded scaler)
    feature_array_for_scaling_scaled = scaler.transform(feature_array_for_scaling)

    # Replace the numerical features in the original feature array with the scaled values
    feature_array[:, [1, 3, 4, 5, 6, 9, 10]] = feature_array_for_scaling_scaled

    # Make prediction
    prediction = model.predict(feature_array)
    
    # Get the disorder name based on the prediction
    disorder_name = disorder_names.get(prediction[0], "Unknown Disorder")
    
    # Return the result
    return render_template('result.html', prediction=prediction[0], disorder_name=disorder_name)
@app.route('/graphs')
def graphs():
    return render_template("graph.html")

if __name__ == "__main__":
    app.run(debug=True)
