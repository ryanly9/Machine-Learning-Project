from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import numpy as np
import pickle

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/measure')
def measure():    
    return render_template('measure.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    age = float(request.form['age'])
    gender = int(request.form['gender'])
    height_cm = float(request.form['height'])  # Height in centimeters from form
    weight_kg = float(request.form['weight'])  # Weight in kilograms from form
    name = str(request.form['name'])  #Name in string from form

    # Convert height from centimeters to meters for BMI calculation
    height_m = height_cm / 100

    # Calculate BMI
    bmi = weight_kg / (height_m ** 2)

    # Get the level of physical activity
    physical_activity = int(request.form['activity'])

    # Prepare the features for the model
    features = [age, gender, height_cm, weight_kg, bmi, physical_activity]
    final_features = np.array(features).reshape(1, -1)

    print("Form Data:",name, age, gender, height_cm, weight_kg, physical_activity)
    print("BMI:", bmi)
    print("Final Features:", final_features)
    
    

    # Predict using the loaded model
    prediction = model.predict(final_features)
    if prediction == 0:
         output = 'Under Weight' 
    if prediction == 1:
         output = 'Normal Weight'
    if prediction == 2:
        output = 'Over Weight'
    if prediction == 3:
        output= 'Obese'

    
    print("Prediction:", output)
    bmi_r = round(bmi,2)
    
    global sheet_data,f #making a global variable for saving data in csv format
    if gender==1:
        gend = "Male"
    
    elif gender == 0:
        gend = "Female"
        
        
    sheet_data = f"{name},{age}, {gend}, {height_cm}, {weight_kg}, {physical_activity},{bmi_r},{output}\n "
    print(sheet_data)
    f = open('records.csv', 'a')
    lines = [sheet_data]
    f.writelines(lines)
    f.close()
    
    
    
   

    return render_template('measure.html', prediction_text= f"Health Status of {name} is {output}", bmi_re= f"{bmi_r} kg/m²",  output =output )



@app.route('/show')
def show():
    return render_template('show.html')
@app.route('/algo')
def algo():
    return render_template('algo.html')
@app.route('/conclusion')
def conclusion():
    return render_template('conclusion.html')
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
    



