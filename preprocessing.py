
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(
    "data/accident_2019.csv",
    dtype={
        "Accident_Index": "string",
        "LSOA_of_Accident_Location": "string"
    },
    low_memory=False
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Display first five records
print(df.head())

# Check target distribution
print("\nAccident Severity Distribution:")
print(df["Accident_Severity"].value_counts().sort_index())


# --------------------------------
# 1. Remove unnecessary columns
# --------------------------------

columns_to_drop = [
    "Accident_Index",
    "LSOA_of_Accident_Location",
    "Did_Police_Officer_Attend_Scene_of_Accident",
    "Number_of_Casualties"
]

df = df.drop(columns=columns_to_drop)

print("\nColumns removed:", columns_to_drop)


# --------------------------------
# 2. Convert Date and extract features
# --------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Day_of_Week"] = df["Date"].dt.day_of_week

# --------------------------------
# 3. Extract hour from Time
# --------------------------------

df["Hour"] = pd.to_datetime(
    df["Time"],
    format="%H:%M",
    errors="coerce"
).dt.hour

# Remove original Date and Time columns
df = df.drop(columns=["Date", "Time"])

print("\nNew dataset shape:", df.shape)

print("\nUpdated columns:")
print(df.columns.tolist())

print("\nMissing values after date/time processing:")
print(df.isnull().sum())


# --------------------------------
# 4. Inspect remaining features
# --------------------------------

columns_to_inspect = [
    "Police_Force",
    "Local_Authority_(District)",
    "Local_Authority_(Highway)",
    "1st_Road_Class",
    "1st_Road_Number",
    "2nd_Road_Class",
    "2nd_Road_Number",
    "Number_of_Vehicles"
]

for col in columns_to_inspect:
    print(f"\n{col}")
    print("Unique values:", df[col].nunique())
    print(df[col].value_counts().head(10))


# --------------------------------
# 5. Initial feature selection
# --------------------------------

categorical_features = [
    "Police_Force",
    "1st_Road_Class",
    "2nd_Road_Class",
    "Road_Type",
    "Junction_Detail",
    "Junction_Control",
    "Pedestrian_Crossing-Human_Control",
    "Pedestrian_Crossing-Physical_Facilities",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Special_Conditions_at_Site",
    "Carriageway_Hazards",
    "Urban_or_Rural_Area",
    "Day_of_Week"
]

numerical_features = [
    "Speed_limit",
    "Month",
    "Day",
    "Hour"
]

target = "Accident_Severity"

# Select input features and target
X = df[categorical_features + numerical_features].copy()
y = df[target].copy()

print("\nFeature selection completed!")
print("Input feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nCategorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))
print("Total input features:", X.shape[1])

print("\nTarget distribution:")
print(y.value_counts().sort_index())
