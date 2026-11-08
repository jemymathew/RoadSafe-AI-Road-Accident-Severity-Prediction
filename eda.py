
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(
    "data/accident_2019.csv",
    dtype={
        "Accident_Index": "string",
        "LSOA_of_Accident_Location": "string"
    },
    low_memory=False
)

# 1. Target distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Accident_Severity",
)

plt.title("Distribution of Accident Severity")
plt.xlabel("Accident Severity (1=Fatal, 2=Serious, 3=Slight)")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

# 2. Target percentage
print("\nSeverity percentages:")
print(
    df["Accident_Severity"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

# 3. Missing value percentage
missing = df.isnull().sum()
missing = missing[missing > 0]

print("\nMissing value percentages:")
print((missing / len(df) * 100).sort_values(ascending=False))

# 4. Numerical feature statistics
print("\nNumerical statistics:")
print(df.describe().T)
