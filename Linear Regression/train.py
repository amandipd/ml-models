import numpy as np
import pandas as pd
from LinearRegression import LinearRegression


data = pd.read_csv('data.csv')
X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values

# Create and train the model
model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X, y)


y_pred = model.predict(X)

# Print the learned parameters
print("Learned weights:", model.weights)
print("Learned bias:", model.bias)
