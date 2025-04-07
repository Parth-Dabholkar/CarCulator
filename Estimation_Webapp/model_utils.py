import joblib
import pandas as pd
from preprocess_utils import clean_data, preprocess

# Load trained model and training columns
model = joblib.load("xgb_model.pkl")
X_columns = joblib.load("X_columns.pkl")

def predict_price(user_input_dict):
    df = pd.DataFrame([user_input_dict])

    # Preprocess
    df = clean_data(df)
    df = preprocess(df)

    # Align user input with training columns
    df = df.reindex(columns=X_columns, fill_value=0)

    # Predict
    prediction = model.predict(df)
    return prediction[0]
