from flask import Flask, render_template, request
import pandas as pd
import time
import joblib
from model_utils import clean_data, preprocess  # Assuming you saved preprocessing in a separate file

app = Flask(__name__)

# Load trained model and feature columns
model = joblib.load("xgb_model.pkl")  # Replace with your actual model path
X_columns = joblib.load("X_columns.pkl")  # Feature columns used during training

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Extract form data
        user_input = {
            'Name': request.form['Name'],
            'Location': request.form['Location'],
            'Year': int(request.form['Year']),
            'Kilometers_Driven': int(request.form['Kilometers_Driven']),
            'Fuel_Type': request.form['Fuel_Type'],
            'Transmission': request.form['Transmission'],
            'Owner_Type': request.form['Owner_Type'],
            'Mileage': request.form['Mileage'],
            'Engine': request.form['Engine'],
            'Power': request.form['Power'],
            'Seats': float(request.form['Seats'])
        }

        df_input = pd.DataFrame([user_input])
        df_input = clean_data(df_input)
        df_input = preprocess(df_input)
        df_input = df_input.reindex(columns=X_columns, fill_value=0)

        time.sleep(5)  # Simulate delay
        pred_price = model.predict(df_input)[0]

        return render_template("index.html", prediction=pred_price)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
