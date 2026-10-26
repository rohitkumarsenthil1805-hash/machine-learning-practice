#nifty 50 stock closing price prediction
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = yf.download("^NSEI", period="2y", auto_adjust=True)

X = df[["Open", "High", "Low", "Volume"]]
y = df["Close"].squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))

latest_data = X.iloc[[-1]]
prediction = model.predict(latest_data)

print("Latest Actual Closing Price:", round(float(y.iloc[-1]), 2))
print("Predicted Closing Price:", round(float(prediction[0]), 2))
