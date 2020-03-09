import numpy as np
import numpy.linalg as linalg
import pandas as pd
import matplotlib.pyplot
import matplotlib.pyplot as plt

# Polynomial Regression

# Section 1.2: Implementation of Polynomial Regression

poly_data = pd.read_csv('polynomial_regression_data.csv')

poly_data.sort_values('x', ascending = True, inplace = True)

x_train = poly_data['x'].values
y_train = poly_data['y'].values

def poly_regression(x, y):

# A function which does feature expansion up to a certain degree for x

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

    plt.figure()
    plt.plot(x_train, y_train, 'bo')
    
    plt.axhline(y=-1, color='k')

    w1 = (get_weight(x_train, y_train, 1))

    xtest1 = get_matrix(x_train, 1)
    ytest1 = xtest1.dot(w1)
    plt.plot(x_train, ytest1, 'r')

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

    plt.xlim((-5, 5))
    plt.legend(('Ground Truth', '$x^0$', '$x$', '$x^2$', '$x^3$', '$x^5$', '$x^{10}$' ))

    plt.savefig('polynomial1.png')

poly_regression(x_train, y_train)

# Section 1.3: Evaluation

def evaluate(x, y):

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

    train_data, test_data = np.split(poly_data, [int(.7*len(poly_data))])

    x_train = train_data['x'].values
    y_train = train_data['y'].values

    x_test = test_data['x'].values
    y_test = test_data['y'].values

    rmse_train = np.zeros((11,1))
    rmse_test = np.zeros((11,1))

    for i in range(1, 12):
        Xtrain = get_matrix(x_train, i)
        Xtest  = get_matrix(x_test, i)
        
        w = get_weight(x_train, y_train, i)
        
        rmse_train[i - 1] = np.sqrt(np.mean((Xtrain.dot(w) - y_train)**2))
        rmse_test[i - 1] = np.sqrt(np.mean((Xtest.dot(w) - y_test)**2))
        
    plt.figure()
    plt.semilogy(range(1,12), rmse_train)
    plt.semilogy(range(1,12), rmse_test)
    plt.legend(('RMSE on Training Set', 'RMSE on Test Set'))

evaluate(x_train, y_train)







