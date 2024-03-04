import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from constants import *

def LogColumn(arr, collist=[0,4]):
    for ind in collist:
        arr[:,ind] =  np.ma.log(arr[:,ind])
    return arr

def MinMaxFind(arr, mask):
    maskedarr = np.ma.masked_equal(arr,mask,copy=False)
    min = np.min(maskedarr)
    max = np.max(maskedarr)
    return min, max

def MinMaxScale(matinput, mask, offset = 0):
    mat = matinput.astype('float32')
    for i in range(np.shape(mat)[1]):
        arr = mat[:,i]
        min, max = MinMaxFind(arr, mask)
        arr = np.where(arr!=mask, (arr-min)/(max-min) + offset, arr)
        arr[np.isnan(arr)] = offset
        mat[:,i] = arr
    return mat

def makeinput(samplefile, id, nobjects):
    df = pd.read_csv(samplefile, engine = 'python', header=None)
    data = df.values.astype('float32')
    nfeatures = int(np.shape(df)[1])
    nsamp = int(np.shape(df)[0]/nobjects)

    loglist = [0,4]
    sampleData = LogColumn(data, loglist)

    sampleData = MinMaxScale(sampleData, maskValue1, 0.0001)
    sampleData[sampleData < -100.0] = np.float32(maskValue2)

    sampleData = sampleData.reshape(nsamp, nobjects, nfeatures)
    sampleLabel = [id]*nsamp

    print("Created Input")
    print(np.shape(sampleData))

    return sampleData, sampleLabel

def create_input(darkjetfile, qcdjetfile, nobjects=100):
    darkSample, darkLabel = makeinput(darkjetfile, 1, nobjects)
    print("Dark Jet sample Processed")
    print("Shape:", np.shape(darkSample), " ", np.shape(darkLabel))

    qcdSample, qcdLabel = makeinput(qcdjetfile, 0, nobjects)
    print("QCD Jet sample Processed")
    print("Shape:", np.shape(qcdSample), " ", np.shape(qcdLabel))

    print("Shapes: ")
    print("Dark and QCD Sample :: ", np.shape(darkSample), np.shape(qcdSample))
    print("Dark and QCD Label  :: ", np.shape(darkLabel), np.shape(qcdLabel))

    data = np.concatenate((darkSample, qcdSample), axis=0)
    dataLabel = np.concatenate((darkLabel, qcdLabel), axis=0)

    data, dataLabel = shuffle(data, dataLabel)

    return data, dataLabel