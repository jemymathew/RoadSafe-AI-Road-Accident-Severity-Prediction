
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "data/accident_2019.csv",
    low_memory=False
)

categorical_features = [
    "Day_of_Week",
    "Road_Type",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Junction_Detail",
    "Junction_Control",
    "Urban_or_Rural_Area"
]

severity_labels = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

df["Severity_Label"] = df["Accident_Severity"].map(
    severity_labels
)

for col in categorical_features:

    # Calculate percentage distribution
    percentage = pd.crosstab(
        df[col],
        df["Severity_Label"],
        normalize="index"
    ) * 100

    percentage = percentage.reindex(
        columns=["Fatal", "Serious", "Slight"]
    )

    print(f"\n{col} vs Accident Severity")
    print(percentage.round(2))

    # Plot stacked bar chart
    percentage.plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5)
    )

    plt.title(f"{col} vs Accident Severity")
    plt.xlabel(col)
    plt.ylabel("Percentage (%)")
    plt.legend(title="Accident Severity")
    plt.tight_layout()
    plt.show()


# --------------------------------
# Sample counts for each category
# --------------------------------

for col in categorical_features:

    print(f"\nSample counts for {col}:")

    counts = pd.crosstab(
        df[col],
        df["Severity_Label"]
    )

    counts["Total"] = counts.sum(axis=1)

    print(counts)
