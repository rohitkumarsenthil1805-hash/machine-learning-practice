#Student Score Prediction using Linear Regression
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([35, 40, 50, 60, 70])
model = LinearRegression()
model.fit(X, y)
pred = model.predict([[6]])
print("Predicted score:", pred[0])
plt.scatter(X, y, color="blue", label="Actual scores")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Student Score Prediction")
plt.legend()
plt.show()
