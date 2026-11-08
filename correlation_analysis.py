
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(
    "data/accident_2019.csv",
    low_memory=False
)

# Select numerical features
numerical_features = [
    "Number_of_Vehicles",
    "Number_of_Casualties",
    "Speed_limit",
    "Location_Easting_OSGR",
    "Location_Northing_OSGR",
    "Longitude",
    "Latitude",
    "Accident_Severity"
]

# Calculate correlation matrix
correlation_matrix = df[numerical_features].corr(
    method="pearson"
)

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))

# Plot heatmap
plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()
