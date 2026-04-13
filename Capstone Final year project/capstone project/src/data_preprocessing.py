import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# URL for the live Google Sheet CSV export
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1y_UlIjIqQvBfugews-9RstOPkZD4BVfIeOaX2aapKZc/export?format=csv"

def load_and_preprocess():
    """
    Loads data from Google Sheets, cleans numeric fields, 
    derives features, and encodes categorical data for ML.
    """
    # 1. Load LIVE data
    try:
        df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

    # 2. Save snapshot for backup
    try:
        if not os.path.exists("../data"):
            os.makedirs("../data")
        df.to_csv("../data/latest_form_data.csv", index=False)
    except Exception:
        pass 

    # 3. Rename columns if necessary (handling potential variations in Form headers)
    # This ensures consistency even if Form questions change slightly
    df.columns = [c.strip() for c in df.columns]

    # 4. Clean numeric columns
    numeric_cols = [
        "Age", "DailyScreenTime(hours)", "AI_Usage_Time(hours/day)", 
        "SleepHours", "PhysicalActivity(hours/day)", "ProductivityScore",
        "Problems_Solved_Before_AI", "Problems_Solved_With_AI"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, force invalid strings to NaN, then fill with median
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Fill NaNs with the median of the column to avoid dropping too much data
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

    # 5. Standardize Categorical Text
    # Specifically StressLevel and MentalHealthImpact logic
    if "StressLevel" in df.columns:
        df["StressLevel_Raw"] = df["StressLevel"].astype(str).str.strip().str.capitalize()

    # 6. Derive Mental Health Impact (Rule-Based)
    def mental_health_logic(row):
        stress = str(row.get("StressLevel", "")).strip().lower()
        sleep = row.get("SleepHours", 8)
        
        if stress == "high" or (stress == "medium" and sleep < 5):
            return "Poor"
        elif stress == "low" and sleep > 7:
            return "Good"
        else:
            return "Neutral"

    df["MentalHealthImpact"] = df.apply(mental_health_logic, axis=1)

    # 7. Encode Categorical Columns for Machine Learning
    # We create a dictionary to ensure consistent mapping
    # instead of letting LabelEncoder pick arbitrary numbers
    
    le = LabelEncoder()
    categorical_to_encode = [
        "Gender", 
        "Primary_AI_Use", 
        "Reliance_On_AI_For_Learning",
        "StressLevel",
        "MentalHealthImpact"
    ]

    for col in categorical_to_encode:
        if col in df.columns:
            # Clean text before encoding
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            df[col] = le.fit_transform(df[col])

    # Ensure everything returned is clean
    return df

