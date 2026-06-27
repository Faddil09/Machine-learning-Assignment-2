# ============================================================
# QUESTION 1: DATA COLLECTION AND PREPROCESSING
# Stock Price Prediction using Apple Stock Dataset
# ============================================================

# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 2. Load Dataset
# ============================================================

file_path = r"C:\Users\MyPc\OneDrive - Universiti Malaya\Documents\Machine Learning\AAPL.csv"

df = pd.read_csv(file_path)

print("="*60)
print("1. DATA COLLECTION")
print("="*60)
print("Dataset successfully loaded.")
print(f"Total rows    : {df.shape[0]}")
print(f"Total columns : {df.shape[1]}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# ============================================================
# 3. Initial Dataset Inspection
# ============================================================

print("\n" + "="*60)
print("2. INITIAL DATASET INSPECTION")
print("="*60)

print("\nDataset Information:")
df.info()

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types Before Cleaning:")
print(df.dtypes)

# ============================================================
# 4. Data Cleaning
# ============================================================

print("\n" + "="*60)
print("3. DATA CLEANING")
print("="*60)

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows Before Cleaning:")
print(df.duplicated().sum())

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort dataset by date
df = df.sort_values('Date')

# Remove duplicate rows
df = df.drop_duplicates()

# Remove missing values
df = df.dropna()

# Reset index
df = df.reset_index(drop=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())

print("\nData Types After Cleaning:")
print(df.dtypes)

print("\nCleaned Dataset Shape:")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ============================================================
# 5. Select Target Variable
# ============================================================

print("\n" + "="*60)
print("4. TARGET VARIABLE SELECTION")
print("="*60)

# Adjusted Close is selected because it considers stock splits and dividends
data = df[['Adj Close']].values

print("Target variable selected: Adj Close")
print("\nFirst 5 original Adj Close values:")
print(data[:5])

# ============================================================
# 6. Data Normalization
# ============================================================

print("\n" + "="*60)
print("5. DATA NORMALIZATION")
print("="*60)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

print("Normalization method: Min-Max Scaling")
print("\nFirst 5 values before normalization:")
print(data[:5])

print("\nFirst 5 values after normalization:")
print(scaled_data[:5])

print("\nNormalization Check:")
print(f"Minimum normalized value : {scaled_data.min():.4f}")
print(f"Maximum normalized value : {scaled_data.max():.4f}")

# ============================================================
# 7. Convert Dataset into RNN Training Format
# ============================================================

print("\n" + "="*60)
print("6. CONVERT DATASET INTO SUITABLE FORMAT FOR TRAINING")
print("="*60)

def create_sequences(dataset, time_steps=60):
    X = []
    y = []

    for i in range(time_steps, len(dataset)):
        X.append(dataset[i-time_steps:i, 0])
        y.append(dataset[i, 0])

    return np.array(X), np.array(y)

time_steps = 60

X, y = create_sequences(scaled_data, time_steps)

print(f"Sliding window size: {time_steps} trading days")
print("\nBefore reshaping:")
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")

# Reshape input for RNN
# RNN requires input shape: (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

print("\nAfter reshaping into RNN format:")
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")

print("\nRNN input format explanation:")
print("X shape = (samples, time steps, features)")
print(f"X shape = ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})")

# ============================================================
# 8. Train-Test Split
# ============================================================

print("\n" + "="*60)
print("7. TRAIN-TEST SPLIT")
print("="*60)

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("Training set: 80%")
print("Testing set : 20%")

print("\nFinal Dataset Shape:")
print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"y_test shape  : {y_test.shape}")

# ============================================================
# 9. Final Summary
# ============================================================

print("\n" + "="*60)
print("8. PREPROCESSING SUMMARY")
print("="*60)

print(f"Total dataset rows              : {len(df)}")
print(f"Missing values after cleaning   : {df.isnull().sum().sum()}")
print(f"Duplicate rows after cleaning   : {df.duplicated().sum()}")
print(f"Target variable                 : Adj Close")
print(f"Normalization method            : Min-Max Scaling")
print(f"Sliding window size             : {time_steps} days")
print(f"Total sequences created         : {len(X)}")
print(f"Training samples                : {len(X_train)}")
print(f"Testing samples                 : {len(X_test)}")

# ============================================================
# QUESTION 3: MODEL DEVELOPMENT
# ============================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

# Create RNN model
model = Sequential()

model.add(SimpleRNN(units=50,
                    return_sequences=False,
                    input_shape=(X_train.shape[1], 1)))

model.add(Dropout(0.2))

model.add(Dense(1))

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

print(model.summary())

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Predict stock prices
y_pred = model.predict(X_test)

# Convert back to original stock price scale
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("="*60)
print("MODEL EVALUATION")
print("="*60)

print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('RNN Training Performance')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14,6))

plt.plot(
    y_test_actual,
    label='Actual Price'
)

plt.plot(
    y_pred_actual,
    label='Predicted Price'
)

plt.title('Actual vs Predicted Apple Stock Price')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)

plt.show()