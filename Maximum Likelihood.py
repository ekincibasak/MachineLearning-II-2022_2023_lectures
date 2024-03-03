import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
%matplotlib inline

# Define training set
X = np.array([-3, -1, 0, 1, 3]).reshape(-1,1) # 5x1 vector, N=5, D=1
y = np.array([-1.2, -0.7, 0.14, 0.67, 1.67]).reshape(-1,1) # 5x1 vector


## EDIT THIS FUNCTION
def max_lik_estimate(X, y):
    
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # returns: maximum likelihood parameters (D x 1)
    
    N, D = X.shape
    
    # Estimate the parameters using the normal equation
    theta_ml = np.linalg.solve(X.T @ X, X.T @ y)
    
    # Alternatively, you can use the cost function to estimate the parameters
    # alpha = 0.01 # Learning rate
    # theta = np.zeros((D, 1)) # Initial parameters
    # for i in range(1000): # Iterate 1000 times
    #     theta = theta - alpha * (X.T @ (X @ theta - y)) / N
    # return theta
    
    return theta_ml


# get maximum likelihood estimate
theta_ml = max_lik_estimate(X,y)

## EDIT THIS FUNCTION
def predict_with_estimate(Xtest, theta):
    
    # Xtest: K x D matrix of test inputs
    # theta: D x 1 vector of parameters
    # returns: prediction of f(Xtest); K x 1 vector
    
    prediction = Xtest @ theta
    
    return prediction 


# define a test set
Xtest = np.linspace(-5,5,100).reshape(-1,1) # 100 x 1 vector of test inputs

# predict the function values at the test points using the maximum likelihood estimator
ml_prediction = predict_with_estimate(Xtest, theta_ml)

# plot
plt.figure()
plt.plot(X, y, '+', markersize=10)
plt.plot(Xtest, ml_prediction)
plt.xlabel("$x$")
plt.ylabel("$y$");







