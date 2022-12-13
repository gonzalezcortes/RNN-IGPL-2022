import sys    
directory = 'Code/'
sys.path.append(directory)
import numpy as np
from ann import models

nn = 10

X_training_lags = np.random.random((3,5,4))
y_training = np.random.random((3,4))
X_test = np.random.random((3,5,4))

epochs_n = 1000
batch_size_n = 64
model = models.model_1()
hist_fit = model.fit(X_training_lags,y_training, epochs = epochs_n, batch_size = batch_size_n, verbose=0)
print(model.predict(X_test))