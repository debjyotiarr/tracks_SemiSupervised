import numpy as np
import plotfunctions as pf

def count_tracks(data):
    #data is reshaped already
    trackCount = [0] * np.shape(data)[0]

    for i in range(np.shape(data)[0]):
        event = data[i,:,0]
        trackCount[i]  = sum(1 for j in event if j > -100.0)

    return trackCount

def multiplicity(data):
    c = count_tracks(data)
    med = np.median(c)

    sigIndex = np.where(c>med)
    bkgIndex = np.where(c<=med)
    predLabel = (c>med).astype(int)
    sigC = [i for i in c if i> med]
    backC = [i for i in c if i<= med]


    m_plot = pf.plot_histogram([sigC, backC],['Signal', 'background'], density=True)
    return sigIndex, bkgIndex, m_plot, predLabel