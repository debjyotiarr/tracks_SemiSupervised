import numpy as np
from constants import *

def stddev_trackfeatures(arr, mask=maskValue2, ax = 0):
    arrnew = np.ma.masked_equal(arr, mask, copy=False)
    stddev = np.std(arrnew, ax)
    return stddev

def get_losses(pred, data):
    loss = np.zeros(np.shape(data)[0])
    error = np.zeros((np.shape(data)[1], np.shape(data)[2]))
    standard_dev = stddev_trackfeatures(data)
    print(standard_dev)
    standard_dev[:,5:13] = 1.0
    print(standard_dev)

    for i in range(np.shape(data)[0]):
        norm = sum(1 for m in data[i, :, 0] if m!= 0.0)
        if i<5:
            print('Norm =',norm)
        for j in range(np.shape(data)[1]):
            for k in range(5):
                #if norm != 0:
                if data[i,j,0] != 0.0:
                    #error[j][k] = np.abs(pred[i][j][k] - data[i][j][k])/norm/standard_dev[j][k]
                    error[j][k] = np.abs(pred[i][j][k] - data[i][j][k])/standard_dev[j][k]
                    loss[i] += error[j][k]
        loss[i] = loss[i]/norm
    return loss