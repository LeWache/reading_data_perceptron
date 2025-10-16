from utilities.reading_folders_with_pattern_and_filter import *
from Figuras import *

saving_folder = "/Users/grte4390/Desktop/Perceptron/Data-Sept/caracterization_27_Sept_25"

root = "/Users/grte4390/Desktop/Perceptron/Data-Sept/caracterization_27_Sept_25"
guide_file = "rate_50.npz"

pattern = r"^rate_(?P<sample_rate>\d+).npz$"

savefig = True

results = find_paths_by_pattern(root, pattern)

all_std = np.zeros(len(results))
all_iteration_time = np.zeros(len(results))
all_int_time = np.zeros(len(results))

for i, r in enumerate(results):
    data = np.load(r)

    sampling_rate = int(1/data["integration_time"])
    Title = f'Current_and_fourier__sampling_rate_{sampling_rate}'
    fig = PlotConfig(ncols=1, nrows=2, Title=Title)

    fig.ax[0,0].plot(data['current']*1e12)
    fig.ax[1,0].plot(data['frqs_fourier'], data['values_fourier'])

    fig.subplot_features(label = [['', 'current (pA)'], ['frequency (1/s)', 'Fourier Transform Value']], add_lbl=False)
    fig.save_figure(saving_folder, fig.Title, savefig=savefig)

    all_std[i] = data['standard_deviation']
    all_iteration_time[i] = data['iterations_time']
    all_int_time[i] = data['integration_time']
#%%
fig = PlotConfig(plot_size=(8,8),ncols=1, nrows=2, Title='Std_and_iteration_time')

fig.ax[0,0].plot(all_int_time, all_iteration_time, **points_kwargs)
fig.ax[1,0].plot(all_int_time, all_std*1e15, **points_kwargs)

fig.subplot_features(label = [['1/sampling rate (s/sample)', 'time one iteration (s)'], ['1/sampling rate (s/sample)', 'standar deviation (fA)']],add_lbl=False)
fig.save_figure(saving_folder, fig.Title, savefig=savefig)