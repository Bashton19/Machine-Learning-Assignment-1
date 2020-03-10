import numpy as np
import numpy.linalg as linalg
import pandas as pd
import matplotlib.pyplot
import matplotlib.pyplot as plt

# Polynomial Regression

# Section 1.2: Implementation of Polynomial Regression

# Reads data in with help of pandas

poly_data = pd.read_csv('polynomial_regression_data.csv')

# Sorts values by x in ascending order

poly_data.sort_values('x', ascending = True, inplace = True)

# Reads x and y columns into numpy arrays

x_train = poly_data['x'].values
y_train = poly_data['y'].values

# Main function which performs polynomial regression
# Takes features_train, y_train, and the degree as arguments

def pol_regression(features_train, y_train, degree):

    # A function which does feature expansion up to a certain degree for x

    def get_matrix(x, degree):
        
        # Creates a matrix of ones in the shape of X
        
        X = np.ones(x.shape)
        
        # For every increment in the degree, stacks columns of arrays
        # This creates a Vandermonde matrix
        
        for i in range(1, degree+1):
            X = np.column_stack((X, x ** i))
        return X 

    # A function which gets weight for each degree

    def get_weight(x, y_train, degree):
        
        # Uses matrix function
        
        X = get_matrix(x, degree)

        # Performs matrix operation to get the vector of estimated
        # polynomial regression coefficients
        # See equation in report

        w = np.mean(y_train)
        XX = X.transpose().dot(X)
        w = np.linalg.solve(XX, X.transpose().dot(y_train))

        return w

    # Creates figure

    plt.figure()
    
    # Plots training points
    
    plt.plot(x_train, y_train, 'bo')
    
    # X^0 is equal to the mean of Y
    
    w0 = np.mean(y_train)
    
    # Gets Vandermonde matrix for 0 degree
    
    xtest0=get_matrix(x_train, 0)
    
    # Computes dot product of Vandermonde matrix and the weight
    # This gives the predicted y value
    
    ytest0 = xtest0.dot(w0)
    
    # Plots regression for x^0 in colour black
    
    plt.plot(x_train, ytest0, 'k')
    
    # Uses weight function for a degree of 1

    w1 = (get_weight(x_train, y_train, 1))

    xtest1 = get_matrix(x_train, 1)
    ytest1 = xtest1.dot(w1)
    plt.plot(x_train, ytest1, 'r')

    # Repeats for weights: 2, 3, 5, 10

    w2 = (get_weight(x_train, y_train, 2))

    xtest2 = get_matrix(x_train, 2)
    ytest2 = xtest2.dot(w2)
    plt.plot(x_train, ytest2, 'y')

    w3 = (get_weight(x_train, y_train, 3))

    xtest3 = get_matrix(x_train, 3)
    ytest3 = xtest3.dot(w3)
    plt.plot(x_train, ytest3, 'g')

    w5 = (get_weight(x_train, y_train, 5))

    xtest5 = get_matrix(x_train, 5)
    ytest5 = xtest5.dot(w5)
    plt.plot(x_train, ytest5, 'm')

    w10 = (get_weight(x_train, y_train, 10))

    xtest10 = get_matrix(x_train, 10)
    ytest10 = xtest10.dot(w10)
    plt.plot(x_train, ytest10, 'c')

    # Sets x axis limit to -5, 5

    plt.xlim((-5, 5))
    
    # Create legend
    
    plt.legend(('ground truth','$x^0$', '$x$', '$x^2$', '$x^3$', '$x^5$', '$x^{10}$'))

# Calls the function

pol_regression(x_train, y_train, 10)

# Section 1.3: Evaluation

# A function which evaluates polynomial regression by comparing the
# root mean square error of training and test data

def eval_pol_regression(features_train, y, degree):
    
    # Redefining matrix and weight functions from above

    def get_matrix(x, degree):
        X = np.ones(x.shape)
        for i in range(1, degree+1):
            X = np.column_stack((X, x ** i))
        return X 

    def get_weight(x, y, degree):
        X = get_matrix(x, degree)

        XX = X.transpose().dot(X)
        w = np.linalg.solve(XX, X.transpose().dot(y))

        return w
    
    # Re-reads data
    
    poly_data_1 = pd.read_csv('polynomial_regression_data.csv')

    # Randomly shuffles data

    poly_data_2 = poly_data_1.sample(frac=1)
    
    # Splits data into 70% training, and 30% test

    train_data = poly_data_2[0:(int(round(len(poly_data_2)*0.7)))]
    test_data = poly_data_2[(int(round(len(poly_data_2)*0.7))):(len(poly_data_2))]

    # Reads x and y of training data into numpy arrays

    x_train = train_data['x'].values
    y_train = train_data['y'].values

    # Reads x and y of test data into numpy arrays

    x_test = test_data['x'].values
    y_test = test_data['y'].values

    # Creates arrays of zeros for RMSE

    rmse_train = np.zeros((degree-1, 1))
    rmse_test = np.zeros((degree-1, 1))

    # Performs polynomial regression on every degree up to 10

    for i in range(1, degree):
        Xtrain = get_matrix(x_train, i)
        Xtest  = get_matrix(x_test, i)
        if i>=1:
            w = get_weight(x_train, y_train, i)
        elif i == 0:
            w = np.mean(y_train)
        
        # Calculates RMSE; see equation in report
        
        rmse_train[i - 1] = np.sqrt(np.mean((Xtrain.dot(w) - y_train)**2))
        rmse_test[i - 1] = np.sqrt(np.mean((Xtest.dot(w) - y_test)**2))
        
    # Creates new figure
        
    plt.figure()

    # Plots training data RMSE
    
    plt.semilogy(range(1,degree), rmse_train)
    
    # Plots test data RMSE
    
    plt.semilogy(range(1,degree), rmse_test)
    
    # Creates legend
    
    plt.legend(('RMSE on Training Set', 'RMSE on Test Set'))

# Calls function to evaluate polynomial regression

eval_pol_regression(x_train, y_train, 10)







