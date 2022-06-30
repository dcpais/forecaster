import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.tsa.api import ExponentialSmoothing

import numpy as np
import pandas as pd

data = pd.read_csv("data.csv")
dates = data["date"]
rev = data["sales"]

start_date = dates[0]
model = ExponentialSmoothing(
    rev, seasonal = "mul",
    trend = "mul", seasonal_periods = 12
).fit( optimized = True)
f_length = len(dates) + 20
forecast = model.predict(0, f_length)

def plot_model():
    
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
    
    
    
    plt.savefig("Fish Rev")
    plt.show()
    
       
plot_model()