from reading_folders_with_pattern_and_filter import *
from Figuras import *
from scipy.optimize import curve_fit

saving_folder = "/Users/grte4390/Desktop/Perceptron/Data-Sept/caracterization_27_Sept_25"

root = "/Users/grte4390/Desktop/Perceptron/Data-Sept/caracterization_27_Sept_25"
guide_file = "rate_50.npz"

pattern = r"^rate_(?P<sample_rate>\d+)$"

savefig = False

meta_data = find_metadata(root, guide_file, main_pattern=pattern)[0]
results = get_files_from_metadata(meta_data, 'pattern')

all_std = np.zeros(len(results))
all_sample_rate = np.zeros(len(results))

print(len(results))