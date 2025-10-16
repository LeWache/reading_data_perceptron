from utilities.reading_folders_with_pattern_and_filter import *
from Figuras import *

filter = {'eta' : '0.05'} #0.5, 0.1, 0.05, 0.01
Title_aux = '__eta_' + filter['eta']


saving_folder = "/Users/grte4390/Desktop/Perceptron/Data-Oct/caracterization_2_Oct_25"

root = "/Users/grte4390/Desktop/Perceptron/Data-Oct/caracterization_2_Oct_25"
guide_file = "figure.png"

main_pattern = r"^w0_(?P<w0>[^_]+)__e_(?P<eta>[^_]+)__m_(?P<method>.+)$"
secondary_pattern = r"^b_(?P<beta>.+)$"

savefig = True