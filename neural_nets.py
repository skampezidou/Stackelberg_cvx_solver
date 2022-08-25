from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.models import model_from_json
import numpy as np
from keras import backend as bc
from keras.constraints import NonNeg
from keras.activations import linear

def ff(x_train, y_train_m, x_test, timestep, loss, learn_rate, T, K, epochs, batch_size, filename):
    model = Sequential()
    model.add(Dense(128, input_dim=x_train.shape[1], activation="tanh"))#'tanh', kernel_constraint=NonNeg()))#, init="uniform"))
    model.add(Dense(y_train_m.shape[1]))#, kernel_constraint=NonNeg()))

    model.compile(loss=loss, optimizer='sgd', metrics=['mse', my_mape_bc])
    model.fit(x_train, y_train_m, epochs=epochs, batch_size=batch_size)

    y_fit = model.predict(x_train, verbose=0)
    y_pred = model.predict(x_test, verbose=0)

    # serialize model to JSON
    model_json = model.to_json()
    with open("Results/"+"model_"+filename+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Results/"+"model_"+filename+".h5")

    return y_fit, y_pred

def load_ff_model(filename, loss):
    json_file = open("Results/" + "model_" + filename + ".json", 'r')
    loaded_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model)
    loaded_model.load_weights("Results/" + "model_" + filename + ".h5")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model


def ff_samples(p, sigma_all, split_point, nom_loads_i, option, K):
    p_arr = np.asarray(p)
    sigma_all_arr = np.asarray(sigma_all)

    if option == 1:
        nom_loads_i.append(np.mean(nom_loads_i))
        nom_loads_i = [nom_loads_i]*K
        nom_loads_i_arr = np.asarray(nom_loads_i)
        p_arr = np.concatenate((p_arr, nom_loads_i_arr), axis=1)

    y = np.asarray(sigma_all_arr)
    x = np.asarray(p_arr)

    x_train = x[:split_point]
    y_train = y[:split_point]
    x_test = x[split_point:]
    y_test = y[split_point:]
    return x_train, x_test, y_train, y_test

def ff_sampling(p, x1):
    y_train = x1[:len(x1)-1]
    y_test = x1[-1]
    x_train = p[:len(p)-1]
    x_test = p[-1]

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    x_test = np.reshape(x_test, (1, x_test.shape[0]))
    y_test = np.reshape(y_test, (1, y_test.shape[0]))

    return x_train, x_test, y_train, y_test


def my_mape_bc(y_true, y_pred):
    diff=bc.abs((y_true-y_pred)/bc.abs(y_true))
    return 100. * bc.mean(diff, axis=-1)

def my_mape(y_true, y_pred):
    diff = [np.abs(x-y)/np.abs(x) for x,y in zip(y_true, y_pred)]
    return 100. * np.mean(diff)
