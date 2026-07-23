import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# File paths
DATA_FILE = r"C:\Users\DELL\Downloads\smart_energy_monitor\data\cleaned_hourly_energy.csv"
MODEL_FILE = r"C:\Users\DELL\Downloads\smart_energy_monitor\models\energy_forecast_model.pkl"

def train_energy_forecaster(data_path, model_path):
    print("⏳ Loading cleaned energy data...")
    df = pd.read_csv(data_path, parse_dates=["Timestamp"])
    df.set_index("Timestamp", inplace=True)
    
    # 1. Feature Engineering (Extract time components from index)
    df['Hour'] = df.index.hour
    df['Day_of_Week'] = df.index.dayofweek
    df['Is_Weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['Month'] = df.index.month
    
    # 2. Lag Feature (24-hour previous load)
    df['Power_Lag_24h'] = df['Global_active_power'].shift(24)
    
    # Drop rows created with NaN values due to 24h lag
    df = df.dropna()
    
    # Features & Target definition
    features = ['Hour', 'Day_of_Week', 'Is_Weekend', 'Month', 'Power_Lag_24h']
    target = 'Global_active_power'
    
    X = df[features]
    y = df[target]
    
    print(f"⏳ Training Random Forest model on {len(X)} samples...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the trained model
    joblib.dump(model, model_path)
    print(f"✅ SUCCESS! Model trained and saved to:\n{model_path}")

if __name__ == "__main__":
    train_energy_forecaster(DATA_FILE, MODEL_FILE)