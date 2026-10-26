#Iris Dataset Analysis Using Pandas and NumPy
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# 0. Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target_names[iris.target]

# 1. Display first and last records
print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# 2. Dataset information
print("\nShape:", df.shape)
print("\nColumns:", df.columns)
print("\nData types:")
print(df.dtypes)

# 3. Descriptive statistics
print("\nStatistics:")
print(df.describe())

print("\nMean:")
print(df.select_dtypes(include=np.number).mean())

print("\nMedian:")
print(df.select_dtypes(include=np.number).median())

print("\nStandard deviation:")
print(df.select_dtypes(include=np.number).std())

# 4. Check missing values and duplicates
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 5. Select columns
print("\nSepal length:")
print(df["sepal length (cm)"].head())

# 6. Filter records
print("\nFlowers with sepal length > 5:")
print(df[df["sepal length (cm)"] > 5])

# 7. Sort records
print("\nSorted by petal length:")
print(df.sort_values(by="petal length (cm)").head())

# 8. Count each species
print("\nSpecies counts:")
print(df["Species"].value_counts())

# 9. Group by species
print("\nAverage measurements by species:")
print(df.groupby("Species").mean(numeric_only=True))

# 10. Correlation
print("\nCorrelation:")
print(df.select_dtypes(include=np.number).corr())

# 11. NumPy operations
print("\nNumPy mean:", np.mean(values))
print("NumPy maximum:", np.max(values))
print("NumPy minimum:", np.min(values))
print("NumPy standard deviation:", np.std(values))

# 12. Add a new column
df["Petal Size"] = np.where(
    df["petal length (cm)"] > 3, "Large", "Small"
)

print("\nDataset with new column:")
print(df.head())

# 13. Save processed dataset
print("\nProcessed dataset saved successfully.")
