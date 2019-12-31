# from https://pythonspot.com/linear-regression/

import matplotlib
#matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

# first step is to load the dataset using Pandas (Data Analysis module) DataFrame allowing for manipulation by
# rows and columns
# two arrays for X(Size) and Y(Price) - we're looking for a correlation between the two !
# data is split into training and test set
# based on the test data we find a best fit line and make predictions 

# Load CSV and columns
df = pd.read_csv("datasets/housing/Housing.csv")

Y = df['price']
X = df['lotsize']

# Change the structure of the array w/o affecting the data
X=X.values.reshape(len(X),1)
Y=Y.values.reshape(len(Y),1)

# Split the data into training/testing sets
X_train = X[:-250]
X_test = X[-250:]

# Split the targets into training/testing sets
Y_train = Y[:-250]
Y_test = Y[-250:]

# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, Y_train)
# Plot outputs
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3)

#plt.show()

# Make an individual prediction of 5000 using the linear regression model 
rows, cols = (2,1)
value = np.array([[5000]*cols]*rows)
value.reshape(len(value),1)
print( str(regr.predict(value)) )
