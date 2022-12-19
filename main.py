import sys
from keras.engine.base_layer import _apply_name_scope_on_model_declaration    
directory = 'src/'
sys.path.append(directory)
import numpy as np
import pandas as pd
from ann import training, model_LSTM_1, model_GRU_1, model_LSTM_2, model_RNN_1
from utils import preprocessing, plotting, metrics

np.random.seed(0)

data_original = pd.read_csv('data/IBEX35.csv')
data = data_original.drop(columns="Date")

data_norma = preprocessing.norma(data)

X, y = preprocessing.get_steps(data_norma['Close'], data_norma['Volume'])
X_train, X_test, y_train, y_test = preprocessing.splitting(X,y, 2482)

##################
X_train_tensors, X_test_tensors, y_train_tensors, y_test_tensors = training.convert_tensor(X_train,X_test,y_train,y_test)
X_train_tensors_final = training.reshape_3d(X_train_tensors, X_train_tensors.shape[0], 1, X_train_tensors.shape[1])
X_test_tensors_final = training.reshape_3d(X_test_tensors, X_test_tensors.shape[0], 1, X_test_tensors.shape[1]) 

####################
num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 8
hidden_n = 2
n_layers = 1
n_class = 1
dropout_prob = 0.2

lstm1 = model_LSTM_1(input_size, n_class, hidden_n, n_layers, dropout_prob)
gru1 = model_GRU_1(input_size, n_class, hidden_n, n_layers, dropout_prob)
lstm2 = model_LSTM_2(input_size, n_class, hidden_n, n_layers, dropout_prob)
rnn1 = model_RNN_1(input_size, n_class, hidden_n, n_layers, dropout_prob)

silent_mode = True
prediction_1, running_time1 = training.training_iteration(num_epochs, learning_rate, lstm1, X_train_tensors_final, X_test_tensors_final, y_train_tensors, silent_mode)
print("MSE LSTM 1:", metrics.mse(y_test, prediction_1), "Running time: ", running_time1)

prediction_2, running_time2 = training.training_iteration(num_epochs, learning_rate, gru1, X_train_tensors_final, X_test_tensors_final, y_train_tensors, silent_mode)
print("MSE GRU 1:",metrics.mse(y_test, prediction_2), "Running time: ", running_time2)

prediction_3, running_time3 = training.training_iteration(num_epochs, learning_rate, lstm2, X_train_tensors_final, X_test_tensors_final, y_train_tensors, silent_mode)
print("MSE LSTM 2:",metrics.mse(y_test, prediction_3), "Running time: ", running_time3)

prediction_4, running_time4 = training.training_iteration(num_epochs, learning_rate, rnn1, X_train_tensors_final, X_test_tensors_final, y_train_tensors, silent_mode)
print("MSE RNN 1:",metrics.mse(y_test, prediction_4), "Running time: ", running_time4)

#plotting.plot_simple_triple(y_test, prediction_1, prediction_2)