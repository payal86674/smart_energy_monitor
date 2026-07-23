import pandas as pd
import numpy as np
import os

def clean_energy_dataset(raw_file_path, output_file_path):
    print("Reading raw dataset...")
    
    # 1. Load data with correct separator and handle '?' as NaN missing values
    df = pd.read_csv(
        raw_file_path, 
        sep=';', 
        low_memory=False, 
        na_values=['?']
    )
    
    print(f"Raw rows loaded: {len(df)}")
    
    # 2. Drop rows with missing values
    df.dropna(inplace=True)
    
    # 3. Create a single combined Datetime index
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.set_index('Timestamp', inplace=True)
    
    # 4. Convert power columns to numeric data types
    cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in cols:
        df[col] = pd.to_numeric(df[col])
        
    print("Resampling minute-level readings into Hourly averages...")
    # 5. Resample from 1-minute intervals to 1-hour intervals to speed up analysis & ML
    hourly_df = df.resample('1h').mean()
    hourly_df.dropna(inplace=True)
    
    # 6. Feature Engineering for ML forecasting
    hourly_df['Hour'] = hourly_df.index.hour
    hourly_df['Day_of_Week'] = hourly_df.index.dayofweek
    hourly_df['Is_Weekend'] = hourly_df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
    hourly_df['Month'] = hourly_df.index.month
    
    # 7. Save cleaned data to CSV
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    hourly_df.to_csv(output_file_path)
    
    print(f"Data cleaned successfully! Saved {len(hourly_df)} hourly records to '{output_file_path}'")

if __name__ == "__main__":
    # Updated to match your exact file path with the double .txt extension
    RAW_PATH = "data/household_power_consumption/household_power_consumption.txt.txt"
    CLEAN_PATH = "data/cleaned_hourly_energy.csv"
    
    clean_energy_dataset(RAW_PATH, CLEAN_PATH)