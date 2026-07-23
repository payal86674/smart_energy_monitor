import pandas as pd

# Direct path to your new dataset
file_path = r"C:\Users\DELL\Downloads\smart_energy_monitor\data\smart_home_energy_consumption_large.csv"

try:
    df = pd.read_csv(file_path, nrows=5)
    print("✅ File loaded successfully!")
    print("\n--- Column Names ---")
    print(list(df.columns))
    print("\n--- First 2 Rows ---")
    print(df.head(2))
except Exception as e:
    print("❌ Error reading file:", e)