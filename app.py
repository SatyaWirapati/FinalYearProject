from flask import Flask, request, render_template, jsonify, redirect, session
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed to use sessions

# Load model
model = load_model('model/employee_attrition_model.h5')

# Temporary storage for preprocessed data
preprocessed_data = pd.DataFrame()

@app.route('/')
def index():
    global preprocessed_data
    employees = []
    if 'file_name' in session and not preprocessed_data.empty:
        employees = preprocessed_data[['ID', 'Name']].to_dict(orient='records')
    return render_template('index.html', employees=employees)

@app.route('/upload', methods=['POST'])
def upload():
    global preprocessed_data

    if 'file' not in request.files:
        return redirect('/')  # Redirect without error if there's no file
    
    file = request.files['file']
    if file.filename == '':
        return redirect('/')  # Redirect without error if there's no file

    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file)
        
        # Preprocessing the data
        cols_to_drop = ["Over18", "EmployeeCount", "EmployeeNumber", "StandardHours"]
        data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # Data Encoding
        columns_to_encode = ['BusinessTravel', 'OverTime', 'Gender', 'MaritalStatus', 'Department', 'EducationField', 'JobRole']
        encoders = {}
        for column in columns_to_encode:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            encoders[column] = le

        # Data Normalization
        X = data.drop(columns=['Attrition', 'ID', 'Name'], errors='ignore')  # Exclude non-numeric columns
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Reattach the ID and Name columns to the scaled data
        preprocessed_data = pd.concat([data[['ID', 'Name']].reset_index(drop=True), X_scaled_df], axis=1)

        # Store the file name in the session to show it in the custom label
        session['file_name'] = file.filename

        return redirect('/')
    
    return redirect('/')  # Redirect without error if there's an invalid file format

@app.route('/clear-file', methods=['POST'])
def clear_file():
    global preprocessed_data
    preprocessed_data = pd.DataFrame()  # Clear the preprocessed data
    session.pop('file_name', None)
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    global preprocessed_data

    employee_input = request.form.get('employee_name')

    if not employee_input:
        response = {'error': 'Please select a valid employee.'}
        return jsonify(response)

    try:
        employee_id_str, employee_name = employee_input.split(' - ')
        employee_id = int(employee_id_str)
    except (ValueError, IndexError):
        response = {'error': 'Invalid employee format.'}
        return jsonify(response)

    # Find the employee based on ID and validate the name
    employee = preprocessed_data[preprocessed_data['ID'] == employee_id]
    if employee.empty or employee.iloc[0]['Name'] != employee_name:
        response = {'error': 'Employee not found.'}
        return jsonify(response)

    employee = employee.iloc[0]
    employee_data = employee.drop(['ID', 'Name']).values.reshape(1, -1)
    employee_data = employee_data.astype(np.float32)

    prediction = model.predict(employee_data)
    result = '<b>Yes</b>' if prediction[0][0] > 0.5 else '<b>No</b>'

    response = {
        'prediction': f'Employee with ID {employee_id} - {employee["Name"]} will resign: {result}'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
