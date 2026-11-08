
import pandas as pd

df = pd.read_csv(
    "data/accident_2019.csv",
    dtype={
        "Accident_Index": "string",
        "LSOA_of_Accident_Location": "string"
    },
    low_memory=False
)

# Check actual accident dates
df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

print("Date range:")
print("Minimum:", df["Date"].min())
print("Maximum:", df["Date"].max())

# Inspect unusual Junction_Control records
print("\nJunction Control = 0:")
print(
    df.loc[
        df["Junction_Control"] == 0,
        ["Accident_Index", "Junction_Control",
         "Junction_Detail", "Accident_Severity"]
    ]
)

# Inspect special values
columns = [
    "Road_Type",
    "Speed_limit",
    "Junction_Detail",
    "Junction_Control",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Urban_or_Rural_Area"
]

print("\nSpecial code counts:")
for col in columns:
    print(f"\n{col}")
    print(df[col].value_counts().sort_index())
