
import pandas as pd

# Load dataset
df = pd.read_csv("data/accident_2019.csv")

# Basic information
print("Dataset Shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

# Target distribution
print("\nAccident Severity Distribution:")
print(df["Accident_Severity"].value_counts().sort_index())

print("\nTarget percentage:")
print(
    df["Accident_Severity"]
    .value_counts(normalize=True)
    .sort_index() * 100
)
