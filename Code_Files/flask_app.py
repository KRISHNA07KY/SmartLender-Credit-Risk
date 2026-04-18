from flask import Flask, request, jsonify, render_template
import os

from predictor import predict as model_predict

# Serve static files and templates from the existing Static folder
app = Flask(__name__, static_folder='Static', template_folder='Static')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        result = model_predict(data)
        return jsonify(result)
    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
