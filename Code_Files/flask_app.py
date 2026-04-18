from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('model_randomforest.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        encode_map = {
            'Gender': {'Male': 1, 'Female': 0},
            'Married': {'Yes': 1, 'No': 0},
            'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
            'Education': {'Graduate': 1, 'Not Graduate': 0},
            'Self_Employed': {'Yes': 1, 'No': 0},
            'Property_Area': {'Urban': 2, 'Rural': 0, 'Semiurban': 1}
        }

        for col, mapping in encode_map.items():
            data[col] = mapping[data[col]]

        input_vector = np.array([[
            data['Gender'], data['Married'], data['Dependents'], data['Education'],
            data['Self_Employed'], data['ApplicantIncome'], data['CoapplicantIncome'],
            data['LoanAmount'], data['Loan_Amount_Term'], data['Credit_History'],
            data['Property_Area']
        ]], dtype=float)

        prediction = model.predict(input_vector)
        result = "Approved" if prediction[0] == 1 else "Rejected"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
