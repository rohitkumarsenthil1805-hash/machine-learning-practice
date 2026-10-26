#Data Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

#Load dataset
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(
    dict(enumerate(iris.target_names))
)


#Line Plot
plt.plot(df.index, df["sepal length (cm)"])
plt.title("Sepal Length Line Plot")
plt.xlabel("Sample Number")
plt.ylabel("Sepal Length (cm)")
plt.show()

# Bar Plot
df.groupby("species")["petal length (cm)"].mean().plot(
    kind="bar"
)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# Histogram
plt.hist(df["sepal length (cm)"], bins=15)
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot
sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="species"
)
plt.title("Sepal Length vs Petal Length")
plt.show()

