from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('D:/WebiSoftTech/POLYNOMIAL REGRESSION/Salary Experience/Salary_Experience.csv') 

# Prepare the model
X = data[['YearsExperience']].values
y = data['Salary'].values
poly_features = PolynomialFeatures(degree=5)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Create the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict_salary():
    try:
        years_experience = float(request.args.get('experience', 0))
        experience_poly = poly_features.transform(np.array([[years_experience]]))
        predicted_salary = poly_model.predict(experience_poly)[0]
        return jsonify({'experience': years_experience, 'predicted_salary': predicted_salary})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    # Ensures the script is executed directly and not as an imported module
    app.run(debug=True)
