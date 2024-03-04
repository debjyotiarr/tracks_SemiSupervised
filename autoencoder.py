import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import LSTM, Dense, Masking, Lambda
from keras.layers import RepeatVector, TimeDistributed

import plotfunctions as pf
import loss as l
from constants import *


def repeatFunction(x):
    # x[0] is decoded at the end
    # x[1] is inputs
    # both have the same shape

    # padding = 1 for actual data in inputs, 0 for 0
    padding = K.cast(K.not_equal(x[1], maskValue2), dtype=K.floatx())
    # if you have zeros for non-padded data, they will lose their backpropagation

    return x[0] * padding


def make_model(model_param, seq_shape, maskVal=maskValue2):
    #seq_shape = (nobjects, nfeatures)
    input = keras.layers.Input(shape=seq_shape)
    mask1 = Masking(mask_value=maskVal, name='Mask_Layer')(input)
    lstm1 = LSTM(model_param[0], activation='tanh', return_sequences=True, name='Enc_First')(mask1)
    lstm1 = LSTM(model_param[1], activation='tanh', return_sequences=False, name='Enc_Last')(lstm1)

    # latent = LSTM(latsize, activation='tanh', name = 'Latent_layer')(lstm1)
    latent = Dense(model_param[2], activation='relu', name='Latent_layer')(lstm1)

    decoded = RepeatVector(seq_shape[0])(latent)
    lstm2 = LSTM(model_param[1], activation='tanh', return_sequences=True, name='Dec_First_Lstm')(decoded)
    lstm2 = LSTM(model_param[0], activation='tanh', return_sequences=True, name='Dec_Second_Lstm')(lstm2)
    # lstm
    dense = TimeDistributed(Dense(seq_shape[1]), name='Output_Dense')(lstm2)
    output = Lambda(repeatFunction, output_shape=seq_shape)([dense, input])

    model = Model(inputs=input, outputs=output)

    adam = keras.optimizers.Adam(learning_rate=model_param[3])
    model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['accuracy'])
    # model.compile(optimizer=adam, loss=custom_loss, metrics=['accuracy'])

    print(model.summary())
    return model


def Autoencoder(data, model_param, weightfile):
    '''
    :param data: Data file used to test the Autoencoder
    :param model_param: Array of model parameters for the autoencoder: [nLSTM1, nLSTM2, nLAT, LRate]
    :param weightfile: File containing weights for the particular autoencoder (a .h5 file)
    :return: Arrays of indices indicating where the signal events (sigIndex) or background events (backIndex) lie;
        the predicted labels (predLabel) and the histogram of the losses plotted (loss_plot)

    '''
    #model_param = [nLSTM1, nLSTM2, nLAT, L_RATE]
    print("Model parameters:", model_param)
    print("Weightfile Loaded:", weightfile)

    model = make_model(model_param, [np.shape(data)[1], np.shape(data)[2]])
    model.load_weights(weightfile)

    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy'])
    print("Compiled Model :: ", model_param)

    pred = model.predict(data, verbose=2)

    print('Data: ', data)
    print('Pred: ', pred)
    losses = l.get_losses(pred, data)
    print('Losses=', losses)
    minloss = np.min(losses)
    maxloss = np.max(losses)
    print('Min, Max :', minloss, maxloss)
    losses = (losses - minloss) / (maxloss - minloss)
    med = np.median(losses)
    print(losses)
    print('Loss Median =', med)

    sigIndex = np.where(losses>med)
    backIndex = np.where(losses<=med)
    predLabel = (losses > med).astype(int)

    # have to write this function - should take in a bunch of arrays and return the histogram
    loss_plot = pf.plot_histogram([sigloss, backloss], ['Signal', 'background'], density=True)

    return sigIndex, backIndex, predLabel, loss_plot

    #remove to another function
    '''
    sigloss = losses[np.where(dataLabel == 1)]
    backloss = losses[np.where(dataLabel == 0)]

    

    plt.figure()
    plt.hist(sigloss, alpha=0.5, label='Signal', density=True)
    plt.hist(backloss, alpha=0.5, label='Background', density=True)
    plt.legend(loc='best')
    plt.savefig('Losses.png')
    plt.close()



    numSigPred = sum(predLabel)
    numSigTrue = sum(dataLabel)
    print('Num of Predicted Signal =', numSigPred)
    print('Num of True Signal =', numSigTrue)

    plot_roc(dataLabel, predLabel, "ROC_Curve")
    '''
