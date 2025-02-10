""" Plotting script for the thesis

This script creates most plots for the thesis and saves them. It is a plotting script made in a rush,
so it is a little messy. I am sorry.
"""


import os
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import numpy as np

subfolders = [
    # ("mpc_a=2_i=5max", "MPC (actions: 2, iters: $\leq 5$)", "MPC (a:2,i:$\leq5$)"),
    # ("fast", "MPC (fast)", "MPC (fast)"),
    ("mpc_a=2_i=5", "MPC (actions: 2, iters: 5)", "MPC (a:2,i:5)"),
    ("diff_a=2_s=1_i=10", "Diffusion policy (actions: 2, iters: 10, samples: 1)", "Diff (a:2,i:10,s:1)"),
    ("diff_a=2_s=100_i=5", "Diffusion policy (actions: 2, iters: 5, samples: 100)", "Diff (a:2,i:5,s:100)"),
    ("diff_a=1_s=25_i=5", "Diffusion policy (actions: 1, iters: 5, samples: 25)", "Diff (a:1,i:5,s:25)"),
    ("diff_a=1_s=1_i=5", "Diffusion policy (actions: 1, iters: 5, samples: 1)", "Diff (a:1,i:5,s:1)"),
    ("diff_a=1_s=1_i=2", "Diffusion policy (actions: 1, iters: 2, samples: 1)", "Diff (a:1,i:2,s:1)"),

    # ("diff_a=2_i=10_s=1007", "Diffusion policy (actions: 1, iters: 10, samples: 1007)", "Diff (a:1,i:10,s:1007)"),
    # ("diff_a=2_s=1_i=5", "Diffusion policy (actions: 2, samples: 1, iters: 5)"), # maybe do not use
]

name = "all"
base_folder = "output/logs/mm/"
SAVE = True

def figsize(scale, height=1.0):
    fig_width_pt = 418.25368                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0 * height           # Aesthetic ratio (you could change this) (I am)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

if SAVE:
    mpl.use('pgf')

    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,               # LaTeX default is 10pt font.
        "font.size": 10,
        "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": figsize(1.1),     # default fig size of 0.9 textwidth
        "pgf.preamble": r"\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc} \usepackage{siunitx}",    # use utf8 fonts becasue your computer can handle it :)
        }
    mpl.rcParams.update(pgf_with_latex)
    import matplotlib.pyplot as plt

state_dict = {}
computing_dict = {}
timing_dict = {}

vel_min = np.inf
vel_max = -np.inf

# collect all data
for (subfolder, label, _) in subfolders:
    state_dict[subfolder] = list()
    computing_dict[subfolder] = list()
    timing_dict[subfolder] = list()

    flight_data = None
    path = os.path.join(base_folder, subfolder)
    for file in os.listdir(path):
        if file.endswith(".npz"):
            flight_data = np.load(os.path.join(path, file), allow_pickle=True)
            points = np.atleast_3d(flight_data['horizon_states'])
            state_dict[subfolder].append(points[..., 0:6, 0].reshape(-1, 6))

            if len(flight_data['t_wall'][1:]) * 0.02 * flight_data['n_actions'] < 15.0:  # collision
                computing_dict[subfolder].append(flight_data['t_wall'][1:] * 1000)  # remove first step as casadi compiles problem
                timing_dict[subfolder].append(len(flight_data['t_wall'][1:]) * 0.02 * flight_data['n_actions']) # scale 50 Hz step to account for controller being called not every step, remove one step as last action from buffer is never used
            else:
                print(f"Dropped run from {label} because T={len(flight_data['t_wall'][1:]) * 0.02 * flight_data['n_actions']}")


    state_dict[subfolder] = np.concatenate(state_dict[subfolder], axis=0)
    computing_dict[subfolder] = np.concatenate(computing_dict[subfolder], axis=0)
    timing_dict[subfolder] = np.array(timing_dict[subfolder])

    vel = np.linalg.norm(state_dict[subfolder][:, 3:6], axis=1)
    vel_min = min(vel_min, vel.min())
    vel_max = max(vel_max, vel.max())

fig, axes = plt.subplots(nrows=len(subfolders), ncols=1, sharex=True, figsize=figsize(1.1, 0.3*len(subfolders)))

cmap = mpl.colormaps['turbo']
normalizer = mpl.colors.Normalize(vel_min, vel_max)
im = mpl.cm.ScalarMappable(cmap=cmap, norm=normalizer)

for ax, (subfolder, label, _) in zip(axes, subfolders):
    ax.add_patch(plt.Circle((1, 0), 0.12, color="gray", alpha=0.5))
    ax.add_patch(plt.Circle((2, 0), 0.12, color="gray", alpha=0.5))

    states = state_dict[subfolder]

    img = ax.scatter(-states[:, 0] + 1, -states[:, 1] + 1, c=np.linalg.norm(states[:, 3:6], axis=1),
                     cmap=cmap, norm=normalizer, s=1, rasterized=True)

    ax.set_aspect('equal', 'box')
    ax.set_yticks([-0.1, 0.0, 0.1])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f m'))
    ax.set_title(label)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f m'))

cbar = fig.colorbar(im, ax=axes, location='right')
# cbar.ax.set_ylabel(state_groups[1][2], rotation=0)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(r'\SI{%.1f}{\meter\per\second}'))
# fig.tight_layout()

if SAVE:
    fig.savefig('output/plots/{}.pgf'.format(name), dpi=300, bbox_inches='tight')
    fig.savefig('output/plots/{}.pdf'.format(name), dpi=300, bbox_inches='tight')

fig = plt.figure()
ax = fig.subplots()
parts = ax.violinplot(computing_dict.values(), showmeans=True, showextrema=True, showmedians=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor('blue' if "MPC" in subfolders[i][1] else 'green')
    # pc.set_edgecolor('blue')
    #pc.set_alpha(0.5)
parts["cmeans"].set_edgecolor(['blue' if "MPC" in subfolder[1] else 'green' for subfolder in subfolders])
parts["cmins"].set_edgecolor("black")
parts["cmins"].set_alpha(0.3)
parts["cmaxes"].set_edgecolor("black")
parts["cmaxes"].set_alpha(0.3)
parts["cbars"].set_edgecolor("black")
parts["cbars"].set_alpha(0.3)
ax.set_xticks(range(1, len(subfolders)+1), labels=[subfolder[2] for subfolder in subfolders], rotation=45, ha='right', rotation_mode='anchor')
ax.set_ylabel("Computation time")
ax.yaxis.set_major_formatter(FormatStrFormatter('%g ms'))
ax.axhline(40.0, linestyle="--", color='gray', alpha=0.25)
ax.axhline(20.0, linestyle="--", color='gray', alpha=0.25)

if SAVE:
    fig.savefig('output/plots/{}.pgf'.format("compute_time"), dpi=300, bbox_inches='tight')
    fig.savefig('output/plots/{}.pdf'.format("compute_time"), dpi=300, bbox_inches='tight')

fig = plt.figure()
ax = fig.subplots()
parts = ax.boxplot(timing_dict.values())
ax.set_ylabel("Track time")
ax.set_xticks(range(1, len(subfolders)+1), labels=[subfolder[2] for subfolder in subfolders], rotation=45, ha='right', rotation_mode='anchor')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f s'))
ax.axhline(3.08, linestyle="--", color='gray', alpha=0.25)  # 3.08 is the closest control step for the planned gate center point

if SAVE:
    fig.savefig('output/plots/{}.pgf'.format("track_time"), dpi=300, bbox_inches='tight')
    fig.savefig('output/plots/{}.pdf'.format("track_time"), dpi=300, bbox_inches='tight')
# show plot
else:
    plt.show()


## MPC
# horizon_inputs: (T, 4, H)
# horizon_outputs: (T, 22, H)

# horizon_states: (T, 16, H + 1)

## Diffusion
# horizon_actions: (T, 13, H)
# horizon_samples: (T, S, 13, H)

# horizon_states: (T, 12, 1)

