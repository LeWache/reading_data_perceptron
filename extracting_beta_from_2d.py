from reading_folders_with_pattern_and_filter import *
from Figuras import *

saving_folder = "/Users/grte4390/Desktop/Perceptron/Data-Sept/caracterization_25_Sept_25"
savefig = True

root = "/Users/grte4390/Desktop/Perceptron/Data-Sept/caracterization_25_Sept_25"
guide_file = "dummy.dat"

pattern = r"^(?P<folder_number>\d+)$"

results = find_metadata(root, guide_file, main_pattern=pattern)

for r in results:
    folder_path = r["path"]

    detuning = np.memmap(folder_path + "/detuning.dat", dtype="float64")
    dummy = np.memmap(folder_path + "/dummy.dat", dtype="float64")
    data = np.memmap(folder_path + "/rigol_voltage.dat", dtype="float64")

    real_data = data.reshape(len(dummy), len(detuning))

    fig = PlotConfig(Title='Measurement')
    fig.plot_2D(real_data, dummy, detuning, label_x='dummy', label_y='detuning')
    fig.save_figure(folder_path, fig.Title, savefig=savefig)