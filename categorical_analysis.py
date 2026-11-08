
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "data/accident_2019.csv",
    dtype={
        "Accident_Index": "string",
        "LSOA_of_Accident_Location": "string"
    },
    low_memory=False
)

categorical_cols = [
    "Day_of_Week",
    "Road_Type",
    "Speed_limit",
    "Junction_Detail",
    "Junction_Control",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Urban_or_Rural_Area"
]

for col in categorical_cols:

    plt.figure(figsize=(9, 4))

    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    print(f"\n{col} distribution:")
    print(df[col].value_counts())
