
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

numerical_columns = [
    "Number_of_Vehicles",
    "Number_of_Casualties",
    "Speed_limit",
    "Location_Easting_OSGR",
    "Location_Northing_OSGR",
    "Longitude",
    "Latitude"
]

print("NUMERICAL FEATURE SUMMARY")
print(df[numerical_columns].describe().T)

print("\nSKEWNESS")
print(df[numerical_columns].skew())

# Histograms
# Histograms
for col in numerical_columns:
    if col in ["Number_of_Vehicles", "Number_of_Casualties"]:
        plt.figure(figsize=(8, 5))
        sns.countplot(
            data=df,
            x=col,
            color="steelblue"
        )
    else:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col].dropna(), bins=30, kde=True)

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Boxplots
for col in numerical_columns:
    plt.figure(figsize=(7, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()
