import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import stats
import time 

class preprocessing:

    def norma(a, zero_condition):
        if zero_condition == True:
            total_fx = lambda data: (a-np.min(data, axis=0))/(np.max(data, axis=0)-np.min(data, axis=0))
            return total_fx(a)
        else:
            total_fx = lambda data: 2*((data-np.min(data, axis=0))/(np.max(data, axis=0)-np.min(data, axis=0)))-1
            return total_fx(a)

    def get_steps(close, volume, t):
        
        close, volume = np.array(close), np.array(volume)
        n=len(close)
        if t == 1:
            data = np.matrix([[volume[i-4],volume[i-3], volume[i-2], volume[i-1], close[i-4], close[i-3], close[i-2], close[i-1], close[i]] for i in range(10,n)])
            X = data[:,[0,1,2,3,4,5,6,7]]
            y = data[:,8]
            return X, y

        if t==2:
            data = np.matrix([[volume[i-4],volume[i-3], volume[i-2], volume[i-1], close[i-4], close[i-3], close[i-2], close[i-1], close[i], close[i+1]] for i in range(10,n-1)])
            X = data[:,[0,1,2,3,4,5,6,7]]
            y = data[:,[8,9]]
            return X, y

        if t==3:
            data = np.matrix([[volume[i-4],volume[i-3], volume[i-2], volume[i-1], close[i-4], close[i-3], close[i-2], close[i-1], close[i], close[i+1], close[i+2]] for i in range(10,n-2)])
            X = data[:,[0,1,2,3,4,5,6,7]]
            y = data[:,[8,9,10]]
            return X, y

    def splitting(X, y, n_X_train):
        X_train = X[:n_X_train]
        X_test = X[n_X_train:]
        y_train = y[:n_X_train]
        y_test = y[n_X_train:]
        return X_train, X_test, y_train, y_test
    
    def normality_test(data):
        alpha = 0.05
        _,t = stats.normaltest(data)
        if t < alpha:  # null hypothesis: x comes from a normal distribution
            print("The null hypothesis can be rejected")
            print("With a value of", t)
            print("It's not normally distributed")
        else:
            print("The null hypothesis cannot be rejected")
            print("With a value of", t)
            print("It's normally distributed")

class plotting:
    def plot_simple(data):
        plt.plot(np.array(data))
        plt.show()
            
    def plot_simple_doble(data1,data2):
        plt.plot(np.array(data1))
        plt.plot(np.array(data2))
        plt.show()
    
    def plot_simple_triple(data1,data2,data3):
        plt.plot(np.array(data1))
        plt.plot(np.array(data2))
        plt.plot(np.array(data3))
        plt.show()

    def plot_multiple(data):
        n_plots = len(data)
        for i in range(n_plots):
            plt.plot(np.array(data[i]))
        plt.plot()

class metrics:
    def mse(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return mean_squared_error(y_true, y_pred)

