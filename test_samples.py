
import pandas as pd

# Load dataset
df = pd.read_csv("data/accident_2019.csv", low_memory=False)

# Derive Month, Day and Hour from original columns
date = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
time = pd.to_datetime(
    df["Time"].astype(str),
    format="%H:%M",
    errors="coerce"
)

df["Month"] = date.dt.month
df["Day"] = date.dt.day
df["Day_of_Week"] = date.dt.day_of_week
df["Hour"] = time.dt.hour

# Features used by the deployed model
features = [
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
    "Day_of_Week",
    "Speed_limit",
    "Month",
    "Day",
    "Hour"
]

# Extract one actual record for each severity
for severity in [1, 2]:

    sample = df[df["Accident_Severity"] == severity].sample(
        n=1,
        random_state=42
    )

    print("\n" + "=" * 50)
    print("ACTUAL SEVERITY:", severity)
    print("=" * 50)

    print(sample[features].to_dict(orient="records")[0])
