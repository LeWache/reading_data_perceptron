from Figuras import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

# tanh model
def tanh_func(x, x0, beta, A, I0):
    return A * np.tanh(beta * (x - x0)) + I0

def read_npz_tree(root_path):
    """
    Walks through b_{x} folders.
    For each trial_{n}.npz:
      - loads beta (initial guess) and index_learnt
      - loads matching current_{n}.npz with detuning (x) and current (y)
      - fits tanh_func to (detuning, current)
      - stores fitted beta and index_learnt
    Returns:
      betas_all: array of fitted betas
      index_all: array of index_learnt
    """
    root = Path(root_path)
    betas_all = []
    index_all = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("b_"):
            continue

        # Look for trial files in this b-folder
        trial_files = sorted(f for f in folder.iterdir() if f.name.startswith("trial_") and f.suffix == ".npz")

        for trial_file in trial_files:
            n_str = trial_file.stem.split("_")[1]  # get "n" from trial_n
            current_file = folder / f"current_{n_str}.npz"
            if not current_file.exists():
                print(f"Warning: {current_file} missing for {trial_file}")
                continue

            # Load trial data
            trial_data = np.load(trial_file, allow_pickle=True)
            if "beta" not in trial_data or "index_learnt" not in trial_data:
                print(f"Warning: {trial_file} missing beta or index_learnt")
                trial_data.close()
                continue

            beta_init = float(np.atleast_1d(trial_data["beta"])[0])
            index_learnt = trial_data["index_learnt"]
            trial_data.close()

            # Load current data
            current_data = np.load(current_file, allow_pickle=True)
            if "detuning" not in current_data or "current" not in current_data:
                print(f"Warning: {current_file} missing detuning or current")
                current_data.close()
                continue

            y_current = np.array(current_data["current"])
            max_x = current_data["detuning"][-1]
            x_detuning = np.linspace(-max_x, max_x, len(y_current))  # np.array(current_data["detuning"])
            current_data.close()

            # seeds: [x0, beta, A, I0]
            p0 = [
                0.0,
                beta_init,
                (y_current[-1] - y_current[0]) / 2.0,
                (y_current[-1] + y_current[0]) / 2.0,
            ]

            try:
                popt, _ = curve_fit(
                    tanh_func,
                    x_detuning,
                    y_current,
                    p0=p0,
                    maxfev=10000,
                )
                fitted_beta = popt[1]
            except Exception as e:
                print(f"Fit failed for {trial_file}: {e}")
                fitted_beta = np.nan

            betas_all.append(fitted_beta)
            index_all.append(index_learnt)

    print(index_all)

    return np.array(betas_all), np.array(index_all)

# def averaging(beta, index_learnt):
#     betas = []
#     idx = []
#     for b in beta:


if __name__ == "__main__":


    w_0 = "2"

    path = f"/Users/grte4390/Desktop/Perceptron/Data/w0_{w_0}__e_0.1__m_1"
    betas, index_learnt = read_npz_tree(path)

    # filter
    mask = (index_learnt < 400) & (betas < 5)
    betas = betas[mask]
    index_learnt = index_learnt[mask]
    Lspeed = 1 / index_learnt

    # ----- bin + average with SciPy -----
    n_bins = 14  # 14 bins (not edges)
    # mean per bin
    bin_means, edges, _ = binned_statistic(betas, Lspeed, statistic="mean", bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # (optional) error bars: SEM = std / sqrt(N) per bin
    bin_std, _, _ = binned_statistic(betas, Lspeed, statistic="std", bins=edges)
    bin_count, _, _ = binned_statistic(betas, Lspeed, statistic="count", bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_sem = bin_std / np.sqrt(bin_count)

    valid = ~np.isnan(bin_means)  # drop empty bins

    # ----- plot -----
    fig = PlotConfig(r'$\omega_0 = $' + f'{w_0}; ' + r'$\eta = 0.1$', label='data')
    plt.plot(betas, Lspeed, **points_kwargs)

    # with error bars (comment this block if you just want points/line)
    plt.errorbar(centers[valid], bin_means[valid],
                 yerr=bin_sem[valid], fmt="ro-", capsize=3, label="binned avg")

    # if you prefer no error bars, use:
    # plt.plot(centers[valid], bin_means[valid], "ro-", label="binned avg")

    fig.features(r'$\beta (a.u.)$', r'$L_{speed}$ (a.u.)')