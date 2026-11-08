
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(
    "data/accident_2019.csv",
    low_memory=False
)

# Target labels
severity_labels = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

df["Severity_Label"] = df["Accident_Severity"].map(
    severity_labels
)

# Numerical features for analysis
numerical_features = [
    "Number_of_Vehicles",
    "Number_of_Casualties",
    "Speed_limit"
]

# --------------------------------
# Boxplots by accident severity
# --------------------------------

for col in numerical_features:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Severity_Label",
        y=col,
        order=["Fatal", "Serious", "Slight"]
    )

    plt.title(f"{col} vs Accident Severity")
    plt.xlabel("Accident Severity")
    plt.ylabel(col)

    plt.tight_layout()
    plt.show()


# --------------------------------
# Mean values by severity
# --------------------------------

print("\nMean numerical values by accident severity:")

print(
    df.groupby("Severity_Label")[numerical_features]
    .mean()
    .round(2)
)

# --------------------------------
# Median values by severity
# --------------------------------

print("\nMedian numerical values by accident severity:")

print(
    df.groupby("Severity_Label")[numerical_features]
    .median()
)
