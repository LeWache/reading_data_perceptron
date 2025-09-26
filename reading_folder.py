from Figuras import *

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

savefig = True

folder = "/Users/grte4390/Desktop/Perceptron/Data/many_trials_low_change"

n = 27
m = len(np.load(os.path.join(folder, "current_1.npz"))["current"])
currents = np.empty((n,m))
trials = np.empty(n)

for i in range(1,n):
    filename = f"current_{i}.npz"
    filepath = os.path.join(folder, filename)

    data = np.load(filepath)

    currents[i,:] = data["current"]
    trials[i] = i

i0, i1, i2, i3 = 1, 21, 19, 26
fig = PlotConfig(ncols=2, nrows=2, Title='sweep_during_trials')

fig.ax[0,0].plot(currents[i0], **line_kwargs)
fig.ax[0,0].text(0.5, 0.15, f'trial {trials[i0]}',transform=fig.ax[0,0].transAxes, fontsize=15)

fig.ax[1,0].plot(currents[i1], **line_kwargs)
fig.ax[1,0].text(0.5, 0.15, f'trial {trials[i1]}',transform=fig.ax[1,0].transAxes, fontsize=15)

fig.ax[0,1].plot(currents[i2], **line_kwargs)
fig.ax[0,1].text(0.5, 0.15, f'trial {trials[i2]}',transform=fig.ax[0,1].transAxes, fontsize=15)

fig.ax[1,1].plot(currents[i3], **line_kwargs)
fig.ax[1,1].text(0.5, 0.15, f'trial {trials[i3]}',transform=fig.ax[1,1].transAxes, fontsize=15)

fig.subplot_features(add_lbl=False)
fig.save_figure(folder, fig.Title, savefig=savefig)