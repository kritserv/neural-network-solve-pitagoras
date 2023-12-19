import numpy as np
from tensorflow import keras
import os
import time

# Generate Data For Training
A = np.random.rand(10000) * 1000
B = np.random.rand(10000) * 1000
C = np.sqrt(A**2 + B**2)

# Define Model
model = keras.models.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(2,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1)
])

# Compile Model
model.compile(optimizer="adam", loss="mse")

# Define Callback to Clear Terminal And Print C After Each Epoch
class ClearAndPrintC(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        time.sleep(0.5)
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
        pred = model.predict([[4, 3]])[0][0]
        print(f'''
        
 Epoch {epoch}
            
      |\\
      | \\ C={format(pred, ".6f")}
      |  \\
 A=4  |   \\
      |____\\
            
        B=3
            
            ''')

# Train Model
model.fit(np.column_stack((A, B)), C, epochs=50, callbacks=[ClearAndPrintC()])
