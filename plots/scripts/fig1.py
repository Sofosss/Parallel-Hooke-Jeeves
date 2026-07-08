import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

WORKERS = np.array([1, 2, 4, 8, 16, 32])
MULTITHREADING = np.array([757.136, 379.730, 190.947, 96.983, 48.675, 24.713])
MPI = np.array([757.136, 379.730, 191.101, 97.065, 48.806, 25.489])

schemes = [MULTITHREADING, MPI]
labels = ["Multithreading", "MPI"]
colors = ["forestgreen",  "darkorange"]

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 13

plt.rcParams['text.color'] = 'white'; plt.rcParams['axes.labelcolor'] = 'white'; plt.rcParams['xtick.color'] = 'white'; plt.rcParams['ytick.color'] = 'white'

# plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'

def linear_scale(values, ticks):
    positions = np.arange(len(ticks))
    scaled = []
    for v in values:
        for i in range(len(ticks)-1):
            if ticks[i] <= v <= ticks[i+1]:
                scaled_val = positions[i] + (v - ticks[i]) / (ticks[i+1] - ticks[i])
                scaled.append(scaled_val)
                break
        else:
            scaled.append(positions[-1])
    return np.array(scaled)

speedups = [d[0]/d for d in schemes]; efficiencies = [(d[0] / d) / WORKERS for d in schemes]

fig, axes = plt.subplots(1, 2, figsize = (17, 5.5), dpi = 1200)

ax1, ax2 = axes

x = np.arange(len(WORKERS))

for speed, c, l in zip(speedups, colors, labels):
    ax1.plot(
        np.arange(len(WORKERS)),
        linear_scale(speed, WORKERS),
        marker = 'o',
        color = c,
        linewidth = 2,
        label = l
    )

ax1.plot(
    np.arange(len(WORKERS)),
    np.arange(len(WORKERS)),
    linestyle = '--',
    color = 'midnightblue',
    linewidth = 2,
    label = 'Ideal Speedup'
)

ax1.set_xlabel('Workers', fontsize = 18, labelpad = 10)
ax1.set_ylabel('Speedup', fontsize = 18)

ax1.set_xticks(np.arange(len(WORKERS)))
ax1.set_xticklabels(WORKERS, fontsize = 18)

ax1.grid(True, axis = 'both', linestyle = '--', linewidth = 0.62, alpha = 0.55)

handles = [
    Line2D([0], [0], color = colors[i], marker = 'o', lw = 2)
    for i in range(len(colors))
]

ax1_legend = ax1.legend(
    handles = handles,
    labels = labels,
    loc = 'lower left',
    frameon = True,
    facecolor = '#1a1a1a',
    edgecolor = 'white',
    ncols = 2,
    framealpha = 1.0,
    shadow = True,
    fontsize = 12.5,
    bbox_to_anchor = (0.005, 0.9)
)

ax1.add_artist(ax1_legend)

ideal_speedup_handle = [
    Line2D([0], [0], color = 'midnightblue', linestyle = '--', lw = 2)
]

ax1.legend(
    handles = ideal_speedup_handle,
    labels = ['Ideal Speedup'],
    loc = 'lower left',
    frameon = True,
    facecolor = '#1a1a1a',
    edgecolor = 'white',
    framealpha = 1.0,
    shadow = True,
    fontsize = 13.5,
    bbox_to_anchor = (0.005, 0.8)
)


for eff, c, l in zip(efficiencies, colors, labels):
    ax2.plot(
        np.arange(len(WORKERS)),
        eff,
        marker = 'o',
        color = c,
        linewidth = 2,
        label = l
    )

ax2.plot(
    np.arange(len(WORKERS)),
    np.ones_like(WORKERS),
    linestyle = '--',
    color = 'midnightblue',
    linewidth = 2,
    label = 'Ideal Efficiency'  
)

ax2.set_xlabel('Workers', fontsize = 18, labelpad = 10)
ax2.set_ylabel('Parallel Efficiency', fontsize = 18)

ax2.set_xticks(np.arange(len(WORKERS)))
ax2.set_xticklabels(WORKERS, fontsize = 18)

ax2.set_yticks(np.linspace(0, 1, 6))
ax2.set_yticklabels(
    [f"{x*100:.0f}%" for x in np.linspace(0, 1, 6)],
    fontsize = 18
)

ax2.set_ylim(0, 1.05)

ax2.grid(True, axis = 'both', linestyle = '--', linewidth = 0.62, alpha = 0.55)

ax2_legend = ax2.legend(
    handles = handles,
    labels = labels,
    loc = 'lower left',
    frameon = True,
    facecolor = '#1a1a1a',
    edgecolor = 'white',
    ncols = 2,
    framealpha = 1.0,
    shadow = True,
    fontsize = 12.5,
    bbox_to_anchor = (0.005, 0.02)
)

ax2.add_artist(ax2_legend)

ideal_efficiency_handle = [
    Line2D([0], [0], color = 'midnightblue', linestyle = '--', lw = 2)
]

ax2.legend(
    handles = ideal_efficiency_handle,
    labels = ['Ideal Efficiency'],
    loc = 'lower left',
    frameon = True,
    facecolor = '#1a1a1a',
    edgecolor = 'white',
    framealpha = 1.0,
    shadow = True,
    fontsize = 13.5,
    bbox_to_anchor = (0.005, 0.12)
)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('white')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "fig1.png")
plt.savefig(fig_path, dpi = 1200, bbox_inches = "tight", transparent = True)