from Figuras import *

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

file_number = 10395
savefig = True

folder = "/Users/grte4390/Desktop/Perceptron/Data/sweeps_temperature/"
folder_path = folder + str(file_number)


def tanh(x, x0, b, A, A0): return A * np.tanh(b * (x - x0)) + A0

detuning = np.memmap(folder_path + "/detuning.dat", dtype="float64")
dummy = np.memmap(folder_path + "/dummy.dat", dtype="float64")
data = np.memmap(folder_path + "/rigol_voltage.dat", dtype="float64")

real_data = data.reshape(len(dummy), len(detuning))
#%%
fig = PlotConfig(Title='Measurement')
fig.plot_2D(real_data, dummy, detuning, label_x = 'dummy', label_y = 'detuning')
fig.save_figure(folder_path, fig.Title, savefig=savefig)

#%%
colors = plt.cm.jet(np.linspace(0, 1, len(dummy) ))

fig = PlotConfig(Title='All_traces')
for i in range(len(dummy)):
    plt.plot(detuning, real_data[i, :], color=colors[i])
fig.features('detuning', 'current', add_lbl=False)
fig.save_figure(folder_path, fig.Title, savefig=savefig)
#%%
current_example = real_data[0]
A_ex = (current_example[-1] - current_example[0])/2
A0_ex = (current_example[-1] + current_example[0])/2

seeds = [0, 1, A_ex, A0_ex]
betas = np.zeros(len(dummy))
for i in range(len(dummy)):
    par, cov = curve_fit(tanh, detuning, real_data[i, :], p0=seeds)
    betas[i] = par[1]

fig = PlotConfig(Title='Example_fit')
plt.plot(detuning, real_data[-1,:]*1e9, label="data")
plt.plot(detuning, tanh(detuning, *par)*1e9, label="fit")
fig.features('detuning', 'current (nA)')
fig.save_figure(folder_path, fig.Title, savefig=savefig)
#%%

def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2)) * (1 / (sigma * np.sqrt(2 * np.pi)))
x = np.linspace(min(betas)*0.9 ,max(betas)*1.1, 100)

fig = PlotConfig(Title='Distribution')
plt.hist(betas, bins=16, density=True)
plt.plot(x, gaussian(x, 1, np.mean(betas), np.std(betas)))

plt.text(.7, .9, r'$\hat{\beta}=$' + f'{np.mean(betas):.2f}', transform=fig.axes[0].transAxes, fontsize=15)
plt.text(.7, .7,r'$\sigma^2=$' + f'{np.var(betas):.3f}', transform=fig.axes[0].transAxes, fontsize=15)
plt.text(.7, .5,r'$\frac{\sigma^2}{\hat{\beta}^2}=$' + f'{np.var(betas)/(np.mean(betas)**2):.2f}', transform=fig.axes[0].transAxes, fontsize=15)

fig.features(r'$\beta$', 'occurrence', add_lbl=False)
fig.save_figure(folder_path, fig.Title, savefig=savefig)