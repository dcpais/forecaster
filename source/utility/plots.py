import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing

import numpy as np
import pandas as pd



def hwplot(xs, ys, **kwargs):
    """
    Create a Holt winters plot
    """
    
    xdates = pd.date_range(start_date, periods = f_length + 1, freq = "M")
    plt.style.use("./vincents.mlpstyle")
    plt.plot(xdates, forecast)
    plt.plot(xdates[:len(dates)], rev)
    
    plt.legend(["Forecast", "Actual Data"], bbox_to_anchor = (0.5, -0.1), ncol = 2)
    plt.title("Fish")
    plt.ylabel("$ (Australian)")
    plt.xlabel(xlabel=None)
    
    xlim = [xdates[0], xdates[-1]]
    ylim = [min(rev) - min(rev) * 0.2, max(rev) + max(rev) *  0.2]
    yrange = ylim[1] - ylim[0]
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    train_date = xdates[int(len(dates) * 0.8)]
    test_date = xdates[len(dates)]
    
    plt.vlines(train_date, ymin = ylim[0], ymax = ylim[1])
    plt.vlines(test_date, ymin = ylim[0], ymax = ylim[1])
    
    plt.text(train_date, ylim[0] + yrange * 0.4, 
             "Training Period", rotation = -90, 
             backgroundcolor = '#e8e8e8', rotation_mode = "anchor")
    plt.text(test_date, ylim[0] + yrange * 0.4, 
             "Testing Period", rotation = -90, 
             backgroundcolor = '#e8e8e8', rotation_mode = "anchor")
    
    #plt.hlines(ylim[0] + 0.1 * yrange, xmin = xlim[0], xmax = train_date)
    #plt.hlines(ylim[0] + 0.1 * yrange, xmin = train_date, xmax = test_date)

    return plt.gcf()