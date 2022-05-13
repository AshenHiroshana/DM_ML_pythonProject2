# https://www.kaggle.com/vafaknm/tesla-stock-price-prediction-using-lstm-nn
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from matplotlib import pyplot as plt


pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 12)

df0 = pd.read_csv('G:\IIT\Subject\DM & ML\CW2\TSLA.csv', date_parser=True)
# df0.index = pd.to_datetime(df0.index)
df0
df = df0.copy()
print(df.head())
print(df.shape)

# Identification and treatment of any Missing Values
print(df.isnull().values.any())
# No Missing Values

# Identification and treatment of duplicates
duplicateRowsDF = df[df.duplicated()]
print(duplicateRowsDF)
#No duplicates


# Check datatypes of the variables1
print(df.dtypes)
# Convert the Date column to DateTime object
df['Date'] = pd.to_datetime(df['Date'])
print(df.dtypes)
print(df.head())


#stock price change in the last one year
plt.show()
ticker = yf.Ticker('TSLA')
tsla_df = ticker.history(period="1y")
tsla_df['Close'].plot(title="TSLA's stock price")
plt.show()


train_data = df[:202].copy()
print(train_data.head())
print(train_data.shape)

test_data = df[202:].copy()
print(test_data.head())
print(test_data.shape)

train_data = train_data.drop(['Date', 'Adj Close'], axis=1)
print(train_data.head())
print(train_data.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
print(train_data)

x_train = []
y_train = []

for i in range(15, train_data.shape[0]):
    x_train.append(train_data[i - 10:i])
    y_train.append(train_data[i, 3])

x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape)
print(y_train.shape)


model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 5)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=100))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=1000, batch_size=32)

test_data = test_data.drop(['Date', 'Adj Close'], axis=1)
print(test_data.head())
print(test_data.shape)

# ----------

scaler = MinMaxScaler()
test_data = scaler.fit_transform(test_data)
print(test_data)

x_test = []
y_test = []

for i in range(15, test_data.shape[0]):
    x_test.append(test_data[i - 10:i])
    y_test.append(test_data[i, 3])
#
x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)

prdc = model.predict(x_test)
print(prdc)

print(scaler.scale_)

scale = 1 / 2.29405135e-03
print(scale)

prdc = prdc * scale
y_test = y_test * scale



plt.figure(figsize=(13, 13))
plt.plot(y_test, color='green', label="Actual")
plt.plot(prdc, color='red', label="Prediction")
plt.xlabel('Time')
plt.legend()
plt.show()
