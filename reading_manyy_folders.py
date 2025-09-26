import os
import re
import numpy as np
from Figuras import *

saving_folder = "/Users/grte4390/Desktop/Perceptron/Data-Sept/caracterization_24_Sept_25"

root = "/Users/grte4390/Desktop/Perceptron/Data-Sept/caracterization_24_Sept_25"
guide_file = "figure.png"

W0 = 0.1

pattern = r"w0_(.*?)__e_(.*?)__m_(.*)"

ncols = 6
nrows = 4

def find_metadata(root_folder, target_filename, pattern):
    results = []
    pattern_w0 = pattern # matches w0_{initial_weight}__e_{learning_rate}__m_{method}
    pattern_b  = r"b_(.*)"  # matches b_{beta}

    for dirpath, dirnames, filenames in os.walk(root_folder):

        if target_filename in filenames:
            parts = dirpath.split(os.sep)  # break folder path into components

            w0 = eta = method = beta = None

            for part in parts:
                m1 = re.match(pattern_w0, part)
                m2 = re.match(pattern_b, part)
                print(m1, m2)
                if m1:
                    w0, eta, method = m1.groups()
                if m2:
                    beta = m2.group(1)

            results.append({
                "path": dirpath, #os.path.join(dirpath, target_filename),
                "w0": w0,
                "eta": eta,
                "method": method,
                "beta": beta
            })

    return results

def _get_files_from_metadata(meta_data, searched_file):

    paths_current = []
    path_to_check = meta_data["path"]

    current_files = [
        f for f in os.listdir(path_to_check)
        if re.match(f"{searched_file}" + r"_\d+\.npz$", f)
    ]

    for f in current_files:
        full_path = os.path.join(path_to_check, f)
        paths_current.append({"path": full_path})

    return paths_current


results = find_metadata(root, guide_file, pattern)
results.pop(0)
all_betas = []



fig = PlotConfig(Title='Current_traces',ncols=ncols, nrows=nrows, sharex=True, sharey=True)

for i, r in enumerate(results):

    nx, ny = counter_axes(i, ncols, nrows)

    w0 = r["w0"]
    eta = r["eta"]
    method = r["method"]
    beta = r["beta"]

    current_files = _get_files_from_metadata(r, 'current')

    for f in current_files:
        path = f["path"]
        data = np.load(path)

        x = data['detuning']
        y = data['current']
        par = data['data_fit']

        all_betas.append(par[1])

        x = np.linspace(x.min(), x.max(), len(y))

        fig.ax[nx, ny].plot(x, y)

    fig.ax[nx, ny].text(0.05, 0.85, r'$\omega_0=$'+f'{w0}',transform=fig.ax[nx, ny].transAxes, fontsize=10)
    fig.ax[nx, ny].text(0.05, 0.75, r'$\eta=$' + f'{eta}', transform=fig.ax[nx, ny].transAxes, fontsize=10)

fig.subplot_features(add_lbl=False)
fig.save_figure(saving_folder, fig.Title)

fig = PlotConfig(Title='Beta_histogram')
plt.hist(all_betas, bins='auto')

plt.text(.7, .9, r'$\hat{\beta}=$' + f'{np.mean(all_betas):.2f}', transform=fig.axes[0].transAxes, fontsize=15)
plt.text(.7, .7,r'$\sigma^2=$' + f'{np.var(all_betas):.3f}', transform=fig.axes[0].transAxes, fontsize=15)
plt.text(.7, .5,r'$\frac{\sigma^2}{\hat{\beta}^2}=$' + f'{np.var(all_betas)/(np.mean(all_betas)**2):.2f}', transform=fig.axes[0].transAxes, fontsize=15)
plt.text(.7, .3,r'$N_{data}=$' + f'{len(all_betas)}', transform=fig.axes[0].transAxes, fontsize=15)


fig.features(r'$\beta$', 'occurrence', add_lbl=False)
fig.save_figure(saving_folder, fig.Title)
#%%

fig1 = PlotConfig(Title='weights', ncols=ncols, nrows=nrows, sharex=True, sharey=False)
fig2 = PlotConfig(Title='errors',ncols=ncols, nrows=nrows, sharex=True, sharey=False)

for i, r in enumerate(results):

    nx, ny = counter_axes(i, ncols, nrows)

    w0 = r["w0"]
    eta = r["eta"]
    method = r["method"]
    beta = r["beta"]

    trial_files = _get_files_from_metadata(r, 'trial')

    n_iterations = 250
    average_weights = np.zeros(n_iterations)
    average_errors = np.zeros(n_iterations)
    n_average = 0

    colors = plt.cm.jet(np.linspace(0, 1, len(trial_files)))

    for j, f in enumerate(trial_files):
        path = f["path"]
        data = np.load(path)

        weights = data['weights']
        errors = data['errors']

        mask = (~np.isnan(weights)) & (np.abs(weights) <= 1.5) & (~np.isnan(errors)) & (np.abs(errors) <= 2)
        weights_plot = weights[mask]
        errors_plot = errors[mask]

        if len(weights[~mask]) == 0:
            n_average += 1
            average_weights += weights_plot
            average_errors += errors_plot

        fig1.ax[nx, ny].plot(weights_plot, color=colors[j], alpha=0.25)
        fig2.ax[nx, ny].plot(errors_plot,  color=colors[j], alpha=0.25)

    fig1.ax[nx, ny].plot(average_weights/ n_average, color = 'black', alpha=1, label='avg weights')
    fig2.ax[nx, ny].plot(average_errors / n_average, color = 'black', alpha=1, label='avg errors')

    fig1.ax[nx, ny].text(0.05, 0.15, r'$\omega_0=$'+f'{w0}',transform=fig1.ax[nx, ny].transAxes, fontsize=10)
    fig1.ax[nx, ny].text(0.05, 0.05, r'$\eta=$' + f'{eta}', transform=fig1.ax[nx, ny].transAxes, fontsize=10)

    fig2.ax[nx, ny].text(0.05, 0.15, r'$\omega_0=$'+f'{w0}',transform=fig2.ax[nx, ny].transAxes, fontsize=10)
    fig2.ax[nx, ny].text(0.05, 0.05, r'$\eta=$' + f'{eta}', transform=fig2.ax[nx, ny].transAxes, fontsize=10)

fig1.subplot_features(add_lbl=False)
fig1.save_figure(saving_folder, fig1.Title)
fig2.subplot_features(add_lbl=False)
fig2.save_figure(saving_folder, fig2.Title)