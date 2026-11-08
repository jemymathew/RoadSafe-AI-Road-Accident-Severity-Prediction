
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "data/accident_2019.csv",
    low_memory=False
)

features = [
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

for col in features:

    # Percentage of severity classes within each category
    ct = pd.crosstab(
        df[col],
        df["Accident_Severity"],
        normalize="index"
    ) * 100

    print(f"\nSeverity distribution by {col}:")
    print(ct.round(2))

    # Visualization
    ct.plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5)
    )

    plt.title(f"Accident Severity by {col}")
    plt.xlabel(col)
    plt.ylabel("Percentage of Accidents")
    plt.legend(
        title="Severity",
        labels=["Fatal", "Serious", "Slight"]
    )

    plt.tight_layout()
    plt.show()
