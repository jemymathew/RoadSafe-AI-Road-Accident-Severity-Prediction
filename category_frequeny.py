
import pandas as pd

df = pd.read_csv(
    "data/accident_2019.csv",
    low_memory=False
)

features = [
    "Road_Type",
    "Speed_limit",
    "Junction_Detail",
    "Junction_Control",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Urban_or_Rural_Area"
]

for col in features:

    print("\n" + "=" * 50)
    print(f"FEATURE: {col}")
    print("=" * 50)

    counts = df[col].value_counts(dropna=False).sort_index()

    result = pd.DataFrame({
        "Count": counts,
        "Percentage": counts / len(df) * 100
    })

    print(result.round(3))
