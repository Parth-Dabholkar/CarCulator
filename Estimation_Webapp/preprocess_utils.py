import pandas as pd

def clean_data(df):
    df = df.copy()

    # Drop unnecessary columns if they exist
    for col in ['Unnamed: 0', 'New_Price']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Handle missing values
    if 'Seats' in df.columns:
        df['Seats'].fillna(df['Seats'].mode()[0], inplace=True)

    if 'Power' in df.columns:
        df['Power'] = df['Power'].str.replace(' bhp', '', regex=False)
        df['Power'] = pd.to_numeric(df['Power'], errors='coerce').fillna(0)

    if 'Engine' in df.columns:
        df['Engine'] = df['Engine'].str.replace(' CC', '', regex=False)
        df['Engine'] = pd.to_numeric(df['Engine'], errors='coerce').fillna(0)

    if 'Mileage' in df.columns:
        df['Mileage'] = df['Mileage'].str.extract(r'(\d+\.\d+|\d+)')
        df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce').fillna(0)

    return df

def preprocess(df):
    df = df.copy()
    df = pd.get_dummies(df, drop_first=True)
    return df
