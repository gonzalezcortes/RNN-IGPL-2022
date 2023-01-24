import numpy as np
import time
np.random.seed(0)

class model_MLP_1:
    def __init__(self, input_size, n_class, n_ni, n_nj, activation):
        self.n_ni, self.n_nj = int(n_ni), int(n_nj)
        self.sw1 = np.matrix(2*np.random.random((input_size,n_ni)) - 1)
        self.sw2 = np.matrix(2*np.random.random((n_ni,n_nj)) - 1)
        self.sw3 = np.matrix(2*np.random.random((n_nj,n_class)) - 1)

        self.act_fx_selection(activation)
        self.first_layer, self.second_layer, self.thrid_layer = None, None, None
        self.delta_1, self.delta_2, self.delta_3 = None, None, None

    def act_fx_selection(self, activation):
        if activation == "sigmoid":
            self.act_fx = lambda x: 1/(1+np.exp(-x))
            self.deriva = lambda x: x*(1-x)
        if activation == "tahn":
            self.act_fx = lambda x: np.tanh(x)
            self.deriva = lambda x: 1.0 - np.tanh(x)**2

    def forward_propagation(self, input_layer):
        self.first_layer = np.asarray(self.act_fx(np.dot(input_layer, self.sw1)))
        self.second_layer = np.asarray(self.act_fx(np.dot(self.first_layer, self.sw2)))
        self.thrid_layer = np.asarray(self.act_fx(np.dot(self.second_layer, self.sw3)))

    def back_propagation(self, targets, alpha):
        error_3 = np.asarray(np.subtract(targets,self.thrid_layer))
        self.delta_3 = np.asarray(error_3 * alpha * self.deriva(self.thrid_layer))
        error_2 = np.asarray(np.dot(self.delta_3,np.transpose(self.sw3)))
        self.delta_2 = error_2 * alpha * self.deriva(self.second_layer)
        error_1 = np.asarray(np.dot(self.delta_2,np.transpose(self.sw2)))
        self.delta_1 = error_1 * alpha * self.deriva(self.first_layer)

    def weight_update(self, input_layer):
        self.sw1 = self.sw1 + np.dot(input_layer.T, self.delta_1)
        self.sw2 = self.sw2 + np.dot(self.first_layer.T, self.delta_2)
        self.sw3 = self.sw3 + np.dot(self.second_layer.T, self.delta_3)

    def training(self, iterations, alpha, X_train, y_train, X_test):
        t0 = time.time()
        inputs = np.matrix(X_train)
        targets = np.matrix(y_train)
        for i in range(iterations):
            input_layer = np.asarray(inputs)      
            self.forward_propagation(input_layer) ## FORWARD PROPAGATION
            self.back_propagation(targets, alpha) ## BACK PROPAGATION
            self.weight_update(input_layer)       ## WEIGHT UPDATE
        prediction = self.test(X_test)
        running_time = time.time()-t0
        return prediction, running_time

    def test(self, input_layer):
        first_layer = self.act_fx(np.dot(input_layer, self.sw1))
        second_layer = self.act_fx(np.dot(first_layer, self.sw2))
        thrid_layer = self.act_fx(np.dot(second_layer, self.sw3))
        return thrid_layer