covariates = ['x'] # Only one independent variable in this situation
y_names = ['smooth', 'oscillating', 'noisy']
activation = 'elu' # Alternatives 'selu', 'softplus', 'sigmoid', 'tanh', 'elu', 'relu'
epochs = 50000
patience = 1000
batch_size = 10000 # the dataset size is 101, so the data is shown to the training algorithm all at once

width = 10
d_o = 0.0
depth = 5

version = 11
if version == 1:
    width = 1
    d_o = 0.0
    depth = 1
elif version == 2:
    width = 4
    d_o = 0.0
    depth = 1
elif version == 3:
    width = 16
    d_o = 0.0
    depth = 1
elif version == 4:
    width = 1
    d_o = 0.0
    depth = 4
elif version == 5:
    width = 4
    d_o = 0.0
    depth = 4
elif version == 6:
    width = 16
    d_o = 0.0
    depth = 4
elif version == 7:
    width = 1
    d_o = 0.0
    depth = 12
elif version == 8:
    width = 4
    d_o = 0.0
    depth = 12
elif version == 9:
    width = 16
    d_o = 0.0
    depth = 12
elif version == 10:
    width = 64
    d_o = 0.0
    depth = 12
elif version == 11:
    width = 64
    d_o = 1E-3
    depth = 12


# dnn_models = [(1, 0)] is essentially a linear-regression model.  There are len(covariates) + 1 + 2 * len(y_names) parameters.  These are the len(covariates) coefficients of the covariates, plus 1 intercept, in the first layer.  The 2 * len(y_names) are the scale and intercepts of the last layer.  Obviously many of these are redundant, but I keep them all to conform to the standard architecture used here
# dnn_models = [[(N, 0)]] is also essentially a linear-regression model.  There are N * (len(covariates) + 1) + N * len(y_names) + 2 * len(y_names).  Here N * len(covariates) are the coefficients of the covariates in the first layer.  The N * 1 are the N intercepts in the first layer.  The N * len(y_names) are the coefficients of the last layer, and the 2 * len(y_names) are the intercepts of the last layer
dnn_model = [(width, d_o) for d in range(depth)]

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback


if depth < 1:
    sys.stderr.write("depth needs to be positive")
    sys.exit(1)

data = pd.read_csv('data.csv')
X = data.loc[:, covariates]

# stop training when loss is 1E-5, to speed up training
class stopAtLossValue(Callback):
    def on_batch_end(self, batch, logs={}):
        if logs.get('loss') <= 1E-5:
            self.model.stop_training = True

fig, axs = plt.subplots(1, len(y_names), sharex = True, sharey = True, figsize = (7.5, 3))

for ind in range(len(y_names)):
    y = data.loc[:, y_names[ind]]

    model = Sequential()
    first_layer = True
    for defn in dnn_model:
        if first_layer:
            model.add(Dense(defn[0], kernel_initializer = 'normal', activation = 'linear', input_shape = (len(covariates),)))
            first_layer = False
        else:
            model.add(Dense(defn[0], kernel_initializer = 'normal', activation = activation))
        if defn[1] != 0:
            model.add(Dropout(defn[1]))
    model.add(Dense(1, kernel_initializer = 'normal', activation = 'linear'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    model.summary()

    history = model.fit(X, y, epochs = epochs, batch_size = batch_size, verbose = 1, callbacks = [stopAtLossValue(), EarlyStopping(monitor = 'loss', mode = 'min', patience = patience, restore_best_weights = True)])

    a = axs[ind]
    a.plot(X[covariates[0]], y, 'k-', label = 'data')
    x_vals = np.arange(-0.1, 1.1, 0.01)
    a.plot(x_vals, model.predict(x_vals), 'r-', label = 'model')
    a.grid()
    a.set_aspect('equal')
    a.set_xlim([-0.1, 1.1])

handles, labels = a.get_legend_handles_labels()
fig.legend(handles, labels, loc = 'upper right')
fig.text(0.5, 0.04, 'x', ha = 'center', va = 'center')
fig.text(0.06, 0.5, 'Output', ha = 'center', va = 'center', rotation = 'vertical')
plt.suptitle('W=' + str(width) + ', d=' + str(round(d_o, 3)) + ', D=' + str(depth))
plt.savefig('result_' + str(width) + '_' + str(round(d_o, 3)) + '_' + str(depth) + '.png', bbox_inches = 'tight')
plt.show()
plt.close()

sys.exit(0)
