import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.models import model_from_json
from keras.layers import LSTM, Dense, Masking, Lambda
from keras.layers import RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
import matplotlib.pyplot as plt

from constants import *
import autoencoder as ae
import preprocessing as prep
import plotfunctions as pf
import multiplicity as m


nLSTM1 = 300
nLSTM2 = 300
nLAT   = 200
LRate  = 0.001

darkjetfile = "DarkPFlow_Jet1_2k.csv"
qcdjetfile  = "QCDPFlow_Jet1_2k.csv"

weightfile  = "weight_300_300_200.h5"


def AE_Classifier():
    model_param = [nLSTM1, nLSTM2, nLAT, LRate]

    # Use as create_input(darkjetfile, qcdjetfile)  - put the files in
    data, dataLabel = prep.create_input(darkjetfile, qcdjetfile)
    print(data)
    print(dataLabel)

    nTracks = m.count_tracks(data)

    mult_plot = plot_histogram(nTracks)
    mult_plot.plot()
    plt.show()



    #Use as sigIndex, backIndex, predLabel, loss_plot = Autoencoder(data, dataLabel,model_param, weightfile)
    #weightfile = "lstmae_" + folder + "_lat" + str(model_param[2]) + ".h5"
    sigIndex, backIndex, predLabel, loss_plot = ae.Autoencoder(data, model_param, weightfile)

    #Use as rocPlot, df = pf.rocplot(dataLabel, predLabel, **kwargs)
    ##rocPlot, df_roc    = pf.rocplot(dataLabel, predLabel, plotLabel="ROC_Curve")
    ##rocPlotLog, df_roc = pf.rocplot(dataLabel, predLabel, plotLabel="ROC_Curve_Log", logPlot = True)

    ##rocPlot.plot()
    ##rocPlotLog.plot()

AE_Classifier()