# A. Build a baseline model (5 marks) 

# Use the Keras library to build a neural network with the following:

# - One hidden layer of 10 nodes, and a ReLU activation function

# - Use the adam optimizer and the mean squared error  as the loss function.

# 1. Randomly split the data into a training and test sets by holding 30% of the data for testing. You can use the train_test_splithelper function from Scikit-learn.

# 2. Train the model on the training data using 50 epochs.

# 3. Evaluate the model on the test data and compute the mean squared error between the predicted concrete strength and the actual concrete strength. You can use the mean_squared_error function from Scikit-learn.

# 4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.

# 5. Report the mean and the standard deviation of the mean squared errors.

# B. Normalize the data (5 marks) 

# Repeat Part A but use a normalized version of the data. Recall that one way to normalize the data is by subtracting the mean from the individual predictors and dividing by the standard deviation.

# C. Increate the number of epochs (5 marks)

# Repeat Part B but use 100 epochs this time for training.

# D. Increase the number of hidden layers (5 marks)

# Repeat part B but use a neural network with the following instead:

# - Three hidden layers, each of 10 nodes and ReLU activation function.


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

Part A
# Load the data
data = pd.read_csv("concrete_data.csv")

# Split the data into input and output variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Create a list to store the mean squared errors
mse_list = []

# Repeat the process 50 times
for i in range(50):
  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  # Build the model
  model = Sequential()
  model.add(Dense(10, activation="relu", input_shape=(8,)))
  model.add(Dense(1))
  model.compile(optimizer=Adam(lr=0.001), loss="mean_squared_error")

  # Train the model
  model.fit(X_train, y_train, epochs=50, verbose=0)

  # Evaluate the model on the test data
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  mse_list.append(mse)

# Calculate the mean and standard deviation of the mean squared errors
mean_mse = np.mean(mse_list)
std_mse = np.std(mse_list)

print("Mean MSE:", mean_mse)
print("Standard Deviation of MSE:", std_mse)



print('Part B')

# Load the data
data = pd.read_csv("concrete_data.csv")

# Split the data into input and output variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize the input variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a list to store the mean squared errors
mse_list = []

# Repeat the process 50 times
for i in range(50):
  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  # Build the model
  model = Sequential()
  model.add(Dense(10, activation="relu", input_shape=(8,)))
  model.add(Dense(1))
  model.compile(optimizer=Adam(lr=0.001), loss="mean_squared_error")

  # Train the model
  model.fit(X_train, y_train, epochs=50, verbose=0)

  # Evaluate the model on the test data
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  mse_list.append(mse)

# Calculate the mean and standard deviation of the mean squared errors
mean_mse = np.mean(mse_list)
std_mse = np.std(mse_list)

print("Mean MSE:", mean_mse)
print("Standard Deviation of MSE:", std_mse)


# Normalize the input variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

print('Part C')


# Load the data
data = pd.read_csv("concrete_data.csv")

# Split the data into input and output variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize the input variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a list to store the mean squared errors
mse_list = []

# Repeat the process 50 times
for i in range(50):
  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  # Build the model
  model = Sequential()
  model.add(Dense(10, activation="relu", input_shape=(8,)))
  model.add(Dense(1))
  model.compile(optimizer=Adam(lr=0.001), loss="mean_squared_error")

  # Train the model
  model.fit(X_train, y_train, epochs=100, verbose=0)

  # Evaluate the model on the test data
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  mse_list.append(mse)

# Calculate the mean and standard deviation of the mean squared errors
mean_mse = np.mean(mse_list)
std_mse = np.std(mse_list)

print("Mean MSE:", mean_mse)
print("Standard Deviation of MSE:", std_mse)

print('part D')


# Load the data
data = pd.read_csv("concrete_data.csv")

# Split the data into input and output variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize the input variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a list to store the mean squared errors
mse_list = []

# Repeat the process 50 times
for i in range(50):
  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  # Build the model
  model = Sequential()
  model.add(Dense(10, activation="relu", input_shape=(8,)))
  model.add(Dense(10, activation="relu"))
  model.add(Dense(10, activation="relu"))
  model.add(Dense(1))
  model.compile(optimizer=Adam(lr=0.001), loss="mean_squared_error")

  # Train the model for 100 epochs
  model.fit(X_train, y_train, epochs=100, verbose=0)

  # Evaluate the model on the test data
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  mse_list.append(mse)

# Calculate the mean and standard deviation of the mean squared errors
mean_mse = np.mean(mse_list)
std_mse = np.std(mse_list)

print("Mean MSE:", mean_mse)
print("Standard Deviation of MSE:", std_mse)
