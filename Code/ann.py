from tensorflow import keras
from keras import Model
from keras.layers import Layer

class models:
    def model_1():

        inputs_x = keras.layers.Input(shape=(5, 4))
        x = keras.layers.LSTM(64, return_sequences=False, return_state=False, activation="relu")(inputs_x)
        x = keras.layers.Dense(64, activation="relu")(x)
        out_model = keras.layers.Dense(1, activation="relu")(x)

        model = keras.Model(inputs=inputs_x, outputs=[out_model])
        model.compile(loss='mse', optimizer='adam')
    
        return model