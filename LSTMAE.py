import numpy
import matplotlib.pyplot as plt
import pandas
import math
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import optimizers
from keras import losses
from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.layers.merge import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
import scipy.interpolate as interpolate

def load_df(trackfile, towerfile, jetfile):
    df_track = pandas.read_csv(trackfile, engine='python', header=None)
    df_tower = pandas.read_csv(towerfile, engine='python', header=None)
    df_jet   = pandas.read_csv(jetfile,   engine='python', header=None)

    return df_track.values.astype('float32'), df_tower.values.astype('float32'), df_jet.values.astype('float32')

def create_input(track, tower, jet, ns1, ns2, lab1, lab2):
    ns = ns1 + ns2
    scaler = RobustScaler()
    dataX = scaler.fit_transform(track)
    dataY = scaler.fit_transform(tower)
    datajet = scaler.fit_transform(jet)

    dataX, dataY = dataX.reshape(ns,50,5), dataY.reshape(ns,50,8)
    dataZ = [lab1]*ns1 + [lab2]*ns2

    return numpy.array(dataX), numpy.array(dataY), numpy.array(datajet), numpy.array(dataZ)

def getdata(DTr, DTw, DJ, BTr, BTw, BJ):
    DTrack, DTower, DJet = load_df(DTr, DTw, DJ)
    BTrack, BTower, BJet = load_df(BTr, BTw, BJ)
    nsamp_D = numpy.shape(DJ)[0]
    nsamp_B = numpy.shape(BJ)[0]

    track = numpy.concatenate((DTrack, BTrack), axis=0)
    tower = numpy.concatenate((DTower, BTower), axis=0)
    jet = numpy.concatenate((DJet, BJet), axis=0)

    dataTr, dataTw, datajet, dataZ = create_input(track, tower, jet, nsamp_D, nsamp_B, 1, 0)

    trainTr, testTr, trainTw, testTw, trainjet, testjet, trainZ, testZ = train_test_split(dataTr, dataTw, datajet, dataZ, test_size=0.4, shuffle=True)
    validateTr, testTr, validateTw, testTw, validatejet, testjet, validateZ, testZ = train_test_split(testTr, testTw, testjet,
                                                                                          testZ, test_size=0.5,
                                                                                          shuffle=True)
    return trainTr, trainTw, trainjet, trainZ, validateTr, validateTw, validatejet, validateZ, testTr, testTw, testjet, testZ


def lstmae():
    timestep = 50
    NF = 13

    input1 = keras.layers.Input(shape=(50,5,)) #for tracks
    input2 = keras.layers.Input(shape=(50,8,))  #for towers
    input3 = keras.layers.Input(shape=(3,)) #for jets

    #encoder
    lstm1 = LSTM(64, activation='relu', return_sequences=True)(input1)
    lstm1 = LSTM(32, activation='relu', return_sequences=False)(lstm1)

    lstm2 = LSTM(64, activation='relu', return_sequences=True)(input2)
    lstm2 = LSTM(32, activation='relu', return_sequences=False)(lstm2)

    merged = keras.layers.concatenate([lstm1, lstm2, input3])

    model = Dense(200)(merged)
    model = Dense(100)(model)
    model = Dense(50)(model)
    model = Dense(100)(model)
    model = Dense(200)(model)

    model = RepeatVector(timestep)(model)

    #decoder

    model = LSTM(32, activation='relu', return_sequences=True)(model)
    model = LSTM(64, activation='relu', return_sequences=True)(model)

    
    model = TimeDistributed(Dense(NF))(model)  # track and tower features

    aemodel = Model(inputs = [input1, input2, input3], outputs = [model])

    adam = keras.optimizers.Adam(learning_rate=0.0005)
    aemodel.compile(optimizer=adam, loss='binary_crossentropy')

    print(aemodel.summary())

    return aemodel

lstmae()