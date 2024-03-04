import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#def plot_histogram():


def plot_roc(dataLabel, predLabel, **kwargs):
    '''
    :param dataLabel: List of true labels
    :param predLabel: List of predicted labels
    :param kwargs: A dict containing 'plotTitle' (str), 'logPlot'(boolean), 'xLabel', 'yLabel' (str)
    :return: a roc plot either linear or log scale and roc plot data in pandas format
    '''

    plotTitle = kwargs.get('plotTitle', None)
    logPlot   = kwargs.get('logPlot', False)   #By default use linear plot
    if logPlot:
        xlab = kwargs.get('xLabel', 'Background Efficiency')
        ylab = kwargs.get('yLabel', 'Signal Efficiency')
    else :
        xlab = kwargs.get('xLabel', 'Signal Efficiency')
        ylab = kwargs.get('ylabel', 'Background Rejection')

    fpos, tpos, thres = roc_curve(dataLabel, predLabel)
    AEauc = auc(fpos, tpos)
    #print('AUC =',AEauc)

    rocdata = list(zip(fpos,tpos))
    df_roc  = pd.DataFrame(rocdata, columns=['fpos', 'tpos'])

    #with open('ROC_AEClassifier.csv','w') as f:
    #    w = csv.writer(f)
    #    w.writerow(['fpos', 'tpos'])
    #    w.writerows(zip(fpos,tpos))

    if logPlot:
        rocplot = plt.figure()
        plt.semilogy(tpos, 1 / fpos, label='Classifier: AUC = {:.4f}'.format(AEauc))
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.xlim([0.001, 1.0])
        plt.grid(True, which="major", color='#666666', linestyle='-')
        plt.title(plotTitle)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    else :
        rocplot = plt.figure()
        plt.plot([0,1],[0,1])
        plt.plot(fpos, tpos, label = 'Classifier: AUC = {:.4f}'.format(AEauc))
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(plotTitle)
        plt.legend(loc='best')

    return rocplot, df_roc