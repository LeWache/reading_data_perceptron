from reading_folders_with_pattern_and_filter import *
from Figuras import *
from scipy.optimize import curve_fit

saving_folder = "/Users/grte4390/Desktop/Perceptron/Data-Sept/caracterization_25_Sept_25"

root = "/Users/grte4390/Desktop/Perceptron/Data-Sept/caracterization_25_Sept_25"
guide_file = "dummy.dat"

pattern = r"^(?P<folder_number>\d+)$"

savefig = True
#%%

filter_A = 5e-11

results = find_metadata(root, guide_file, main_pattern=pattern)

def try_to_fit(function, x, y, **kwargs_fit):
    try:
        p, c = curve_fit(function, x, y, **kwargs_fit)
        return p, c
    except RuntimeError as e:
        print("Fit failed:", e)
        return None

def filter_currents(current, detuning, amplitud_filter, n_help:int =3, n_std=5):

    # return current, detuning

    amplitud = ( current.max() - current.min() ) / 2
    if amplitud > amplitud_filter:

        idx_min = np.argmin(current)
        if idx_min < len(current)*0.9:

            new_current = current[idx_min:]
            new_detuning = detuning[idx_min:]

            idx_switch = np.argmax(np.diff(new_current))

            first_half = new_current[:idx_switch - n_help]
            last_half = new_current[idx_switch + n_help:]

            if first_half.size>10 and last_half.size>10:

                # condition1 = np.abs(first_half.max() - first_half.min()) < n_std * np.std(first_half)
                # condition2 = np.abs(last_half.max() - last_half.min())   < n_std * np.std(last_half)

                condition1 = abs(first_half[-1] - first_half[0]) <= n_std * np.std(first_half)
                condition2 = abs(last_half[-1] - last_half[0])   <= n_std * np.std(last_half)

                if condition1 and condition2:
                    return new_current, new_detuning

    return None, None

def tanh(x, x0, b, A, A0): return A * np.tanh(b * (x - x0)) + A0

for r in results:
    folder_path = r["path"]
    n = r["folder_number"]

    detuning = np.memmap(folder_path + "/arbitrary_origin_vector.dat", dtype="float64")
    dummy = np.memmap(folder_path + "/dummy.dat", dtype="float64")
    data = np.memmap(folder_path + "/rigol_voltage.dat", dtype="float64")

    real_data = data.reshape(len(dummy), len(detuning))

    if True: #2.5 < detuning[-1] < 3.5:

        fig = PlotConfig(Title=f'Measurement_{n}_filtered', ncols=2, nrows=3, ft=10)
        fig.merge_subplots([[0,0],[0,1]])
        fig.merge_subplots([[2, 0], [2, 1]])

        fig.plot_2D(real_data, dummy, detuning, label_x='', label_y='')

        all_betas = []
        first_filtered = not None

        for i in range(len(dummy)):
            current = real_data[i,:]

            new_current, new_detuning = filter_currents(current, detuning, filter_A)

            if new_current is not None:

                seed_A = (new_current[-1] - new_current[0]) / 2
                seed_A0 = (new_current[-1] + new_current[0]) / 2
                #seeds = [1, 10, seed_A, seed_A0]

                p_results, c_results = try_to_fit(tanh, new_current, new_detuning)
                if results is not None:
                    all_betas.append(p_results[1])

                fig.ax[1,1].plot(new_detuning, new_current)

                if first_filtered is not None:
                    fig.ax[1, 0].plot(new_detuning, new_current)
                    first_filtered = None

        print('length all betas:', len(all_betas))
        print(all_betas)
        fig.ax[2,0].hist(all_betas)

        fig.subplot_features(add_lbl=False)
        fig.save_figure(saving_folder, fig.Title, savefig=savefig)
