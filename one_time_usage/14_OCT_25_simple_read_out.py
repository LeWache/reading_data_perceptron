import matplotlib.pyplot as plt
from Figuras import *
from scipy.optimize import curve_fit
from sympy.printing.pretty.pretty_symbology import line_width

root = "/Users/grte4390/Desktop/Perceptron/Data-Oct/caracterization_15_Oct"

file_names_detuning = []
for i in range(50):
    file_names_detuning.append(f'gross_measurement_{int(i)}.npz')

max_detuning = 8
filter_det = 5
masking_filter = 0.7

def tanh(x, x0, beta):
    return np.tanh(beta * (x - x0))

def linear(x, x0, beta):
    return beta * (x-x0)

colors = plt.cm.inferno(np.linspace(0, 1, len(file_names_detuning)))
fig = PlotConfig()

all_betas_l = np.empty(len(file_names_detuning))
all_betas_t = np.empty(len(file_names_detuning))

for i, name in enumerate(file_names_detuning):
    data = np.load(os.path.join(root, name), allow_pickle=True)

    current  = data['current']
    detuning = data['detuning'] + 2.3

    if current[0] > current[-1]:
        current = current[::-1]

    I_max = np.mean(current[np.where( (detuning > filter_det) & (max_detuning > detuning) )])
    I_min = np.mean(current[np.where( (detuning < -filter_det) & (-max_detuning < detuning) )])

    I_0 = (I_max + I_min)/2
    A   = (I_max - I_min)/2

    normalized_current = (current - I_0) / A

    mask = np.where( np.abs(normalized_current) < masking_filter)

    linearized_current = np.arctanh(normalized_current[mask])
    linearized_detuning = detuning[mask]

    par_t, cov_t = curve_fit(tanh, detuning, normalized_current)
    par_l, cov_l = curve_fit(linear, linearized_detuning, linearized_current)

    beta_t = par_t[1]
    beta_l = par_l[1]

    mask_plot = np.where( (detuning < max_detuning) & (detuning > -max_detuning))

    plt.plot(linearized_detuning, np.tanh(linearized_current), color=colors[i], alpha=0.9, linewidth=5, label=name)
    plt.plot(detuning[mask_plot], normalized_current[mask_plot], color=colors[i], alpha=0.6, linewidth=3)

    print(f'beta tanh = {beta_t:.2f}, beta linear = {beta_l:.2f}, for {name}')

    all_betas_l[i] = beta_l
    all_betas_t[i] = beta_t

fig.features('detuning', 'current', add_lbl=True)

fig = PlotConfig()

plt.plot(all_betas_l, color = 'blue', label = 'fit lineal')
plt.plot(all_betas_t, color = 'red', label = 'fit tanh')

fig.features('iteration', r'$\beta$')
