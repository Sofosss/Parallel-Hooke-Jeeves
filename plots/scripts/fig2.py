import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MPI_RANKS = np.array([1, 2, 4, 8, 16, 32])
ONE_THREAD = np.array([757.136, 383.677, 192.987, 98.185, 49.717, 25.922])
TWO_THREADS = np.array([388.010, 193.229, 98.395, 49.488, 25.755, 14.106])
FOUR_THREADS = np.array([193.488, 98.095, 49.587, 25.849, 14.450, 9.753])

schemes = [ONE_THREAD, TWO_THREADS, FOUR_THREADS]
labels = ["1 thread per MPI process", "2 threads per MPI process", "4 threads per MPI process"]
colors = ["forestgreen",  "darkorange", "dodgerblue"]

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 21
plt.rcParams['ytick.labelsize'] = 21
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.title_fontsize'] = 13

plt.rcParams['text.color'] = 'white'; plt.rcParams['axes.labelcolor'] = 'white'; plt.rcParams['xtick.color'] = 'white'; plt.rcParams['ytick.color'] = 'white'

# plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'

speedups = [ONE_THREAD[0]/d for d in schemes]

fig, ax = plt.subplots(figsize = (9.5, 4.8), dpi = 1200)

for sp, c, l in zip(speedups, colors, labels):
    ax.plot(np.arange(len(MPI_RANKS)), sp, marker = 'o', color = c, label = l, linewidth = 2)

ax.set_xlabel('MPI Ranks', labelpad = 10); ax.set_ylabel('Speedup', labelpad = 10)
ax.set_xticks(np.arange(len(MPI_RANKS))); ax.set_xticklabels(MPI_RANKS)
ax.tick_params(axis = 'y')

handles = [Line2D([0], [0], color=colors[i], marker='o', lw=2) for i in range(len(colors))]
ax.legend(handles = handles, labels = labels,
                           loc = 'upper left', frameon = True, 
                           facecolor = '#1a1a1a', 
                           edgecolor = 'white',
                           ncols = 1, framealpha = 1.0, shadow = True, labelspacing = 0.45, handletextpad = 0.55, 
                           columnspacing = 1.02, bbox_to_anchor = (0.004,  1.005),  borderpad = 0.45)

ax.grid(True, axis = 'both', linestyle = '--', linewidth = 0.62, alpha = 0.55, zorder = 0)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('white')

ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()

fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "fig2.png")
plt.savefig(fig_path, dpi = 1200, bbox_inches = "tight", transparent = True)