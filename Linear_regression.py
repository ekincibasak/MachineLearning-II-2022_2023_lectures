#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


# Importing necessary libraries/modules: numpy for mathematical operations, matplotlib for visualization, train_test_split from sklearn.model_selection for splitting data into training and testing sets, and datasets from sklearn for generating synthetic datasets.
# 

# In[4]:


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Defining two evaluation metrics for the regression model: r2_score() and mean_squared_error(). The former calculates the coefficient of determination, also known as R-squared, while the latter calculates the mean squared error.
# 

# In[5]:


class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated


# Defining a LinearRegression class that implements the gradient descent algorithm for linear regression. The fit() method trains the model on the input data and updates the weights and bias according to the computed gradients. The predict() method predicts the output for new input data based on the learned weights and bias.
# 

# In[6]:


# Generate data
X, y = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)


# Generating a synthetic dataset using datasets.make_regression() and splitting it into training and testing sets using train_test_split().
# 

# In[7]:


# Train model
regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)


# Instantiating the LinearRegression class with a learning rate of 0.01 and maximum number of iterations of 1000, and training the model on the training set using fit(). Then, predicting the output for the testing set using predict().
# 

# In[9]:


# Evaluate model
mse = mean_squared_error(y_test, predictions)
accu = r2_score(y_test, predictions)
print("MSE:", mse)
print("Accuracy:", accu)


# In[13]:


# Plot results
y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()


# In[ ]:




