import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score


df = pd.read_csv("archive/coin_USDCoin.csv")

df["Date"] = pd.to_datetime(df["Date"])

df.drop(["SNo", "Name", "Symbol", "Volume"], axis = 1, inplace = True)

df.sort_values(by = "Date")

df = df[["Date", "Close"]]
plt.figure(figsize = (10, 6))
plt.plot(df["Date"], df["Close"], linestyle = "-")
plt.title("Price over time")
plt.xlabel("Year")
plt.ylabel("Price")
plt.grid(True)
plt.show()

#Data featuring
df["Previous_close"] = df["Close"].shift(1)
df["7_days"] = df["Close"].rolling(window = 7).mean()
df["14_days"] = df["Close"].rolling(window = 14).mean()
df["Change_price"] = df["Close"] - df["Previous_close"]
df["Days_of_week"] = df["Date"].dt.day_of_week
df = df.dropna()

#model
x = df[["Previous_close", "7_days", "14_days", "Change_price", "Days_of_week"]]
y = df["Close"]

poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.4)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = Ridge(alpha = 1.0)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mse = mean_squared_error(y_test, prediction)
print(mse)

cvs = cross_val_score(model, X_poly, y, cv = 5, scoring = "neg_mean_squared_error")

mean_cross = -cvs.mean()
print(mean_cross)

#Plot for actual vs predicted
plt.figure(figsize = (10, 6))
plt.plot(y_test.reset_index(drop = True), label = "Actual price", linestyle = '-')
plt.plot(prediction, label = 'Predicted price', linestyle = '-')
plt.xlabel("Test data point")
plt.ylabel("Price")
plt.title("Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

future_dates = pd.date_range(start = df["Date"].max(), end = "2025-01-01", freq = "D")
last_close = df["Close"].iloc[-1]
future_df = pd.DataFrame(index = future_dates)
future_df["Previous_close"] = [last_close] + [None]*(len(future_dates)-1)
future_df["7_days"] = df["Close"].rolling(window = 7).mean().iloc[-1]
future_df["14_days"] = df["Close"].rolling(window = 14).mean().iloc[-1]
future_df["Change_price"] =  0 
future_df["Days_of_week"] = future_df.index.day_of_week
future_df.fillna(method = "ffill", inplace = True)

#future model
future_poly = poly.transform(future_df)
future_scale = sc.transform(future_poly)
future_prices = model.predict(future_scale)


plt.figure(figsize = (10, 6))
plt.plot(df["Date"], df["Close"], color = "purple", label = "Actual prices")
plt.plot(future_dates, future_prices, color = "green", linestyle = "-", linewidth = 2, label = "Future prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Price prediction")
plt.legend()
plt.grid(True)
plt.show()