import pandas as pd

# File paths
input_path = r"C:\Users\DELL\Downloads\smart_energy_monitor\data\smart_home_energy_consumption_large.csv"
output_path = r"C:\Users\DELL\Downloads\smart_energy_monitor\data\cleaned_hourly_energy.csv"

print("⏳ Step 1: Loading raw dataset...")
df = pd.read_csv(input_path)

print("⏳ Step 2: Combining Date and Time into a single Timestamp...")
# Combine 'Date' and 'Time' columns into a single datetime column
df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df.set_index('Timestamp', inplace=True)

print("⏳ Step 3: Aggregating energy consumption into hourly total power...")
# Group by timestamp to calculate total global energy power per hour
df_hourly = df.groupby('Timestamp').agg({
    'Energy Consumption (kWh)': 'sum',
    'Outdoor Temperature (°C)': 'mean'
}).rename(columns={'Energy Consumption (kWh)': 'Global_active_power'})

# Resample to hourly frequency and forward-fill any empty gaps
df_hourly = df_hourly.resample('1h').mean().ffill()

# Extract appliance sub-metering data if available
if 'Appliance Type' in df.columns:
    print("⏳ Step 4: Extracting appliance sub-metering data...")
    pivoted = df.pivot_table(
        index='Timestamp', 
        columns='Appliance Type', 
        values='Energy Consumption (kWh)', 
        aggfunc='sum'
    ).fillna(0).resample('1h').mean()
    
    # Map appliances to Sub_metering variables used in app.py
    df_hourly['Sub_metering_1'] = pivoted.get('Oven', 0)
    df_hourly['Sub_metering_2'] = pivoted.get('Fridge', 0)
    df_hourly['Sub_metering_3'] = pivoted.get('HVAC', pivoted.get('Heater', 0))

print("⏳ Step 5: Saving cleaned data...")
df_hourly.to_csv(output_path)

print(f"\n✅ SUCCESS! Cleaned dataset saved to:\n{output_path}")
print("\nPreview of processed data:")
print(df_hourly.head())