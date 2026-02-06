import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scores.conformal_utils import *

def plot_classes(X, Y, save_plots_flag = True, res_path = None, show = False):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[Y==0, 0], X[Y==0, 1], c="blue", s=10, label = "0", alpha=0.6)
    plt.scatter(X[Y==1, 0], X[Y==1, 1], c="red", s=10, label = "1", alpha=0.6)
    plt.xlabel("X1", fontsize = 18)
    plt.ylabel("X2", fontsize = 18)
    plt.legend()
    plt.tight_layout()
    if save_plots_flag and res_path!=None:
        plt.savefig(res_path+"plots/synthetic_classes.png", dpi = 200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_score(Xts, tau0ts, tau1ts, rule_limits, changeclsidx, wrong_1_ts, wrong_0_ts, save_plots_flag = True, score_fn = None, res_path = None, show = False):
    norm = colors.Normalize(vmin=0, vmax=1)

    # Plotting for tau0cal
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14, 6))

    for r in range(rule_limits.shape[0]):
        if r<changeclsidx-1:
            ax1.add_patch(plt.Rectangle((rule_limits[r,0], rule_limits[r,2]), abs(rule_limits[r,1]-rule_limits[r,0]), abs(rule_limits[r,3]-rule_limits[r,2]), linewidth=2, edgecolor='tab:green', linestyle='--', facecolor='none'))
            ax2.add_patch(plt.Rectangle((rule_limits[r,0], rule_limits[r,2]), abs(rule_limits[r,1]-rule_limits[r,0]), abs(rule_limits[r,3]-rule_limits[r,2]), linewidth=2, edgecolor='tab:green', linestyle='--', facecolor='none'))
        else:
            ax1.add_patch(plt.Rectangle((rule_limits[r,0], rule_limits[r,2]), abs(rule_limits[r,1]-rule_limits[r,0]), abs(rule_limits[r,3]-rule_limits[r,2]), linewidth=2, edgecolor='orange', linestyle='--', facecolor='none'))
            ax2.add_patch(plt.Rectangle((rule_limits[r,0], rule_limits[r,2]), abs(rule_limits[r,1]-rule_limits[r,0]), abs(rule_limits[r,3]-rule_limits[r,2]), linewidth=2, edgecolor='orange',linestyle='--', facecolor='none'))

    sc = ax1.scatter(Xts[:, 0], Xts[:, 1], c=tau0ts, cmap='Blues', norm = norm, s=120, edgecolors='black', linewidth=1.5, marker='o')

    ax1.scatter(wrong_1_ts[:20, 0], wrong_1_ts[:20, 1], c='red', s=40, marker='x', linewidth=2)

    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('Score (class 0)', fontsize=16)
        #cbar.set_ticks([0, 1])
        #cbar.set_ticklabels(['0.0', '1.0'])
    ax1.set_xlabel("X1", fontsize=18)
    ax1.set_ylabel("X2", fontsize=18)
    ax1.tick_params(labelsize=16)


    sc = ax2.scatter(Xts[:, 0], Xts[:, 1], c= tau1ts, cmap='Blues',  norm = norm, s=120, edgecolors='black', linewidth=1.5, marker='o')
    ax2.scatter(wrong_0_ts[:20, 0], wrong_0_ts[:20, 1], c='red', s=40, marker='x', linewidth=2)


    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('Score (class 1)', fontsize=16)

    ax2.set_xlabel("X1", fontsize=18)
    ax2.set_ylabel("X2", fontsize=18)
    ax2.tick_params(labelsize=16)
    if save_plots_flag and res_path!=None:
        plt.savefig(res_path+f"plots/test_scores_{score_fn}.png", dpi = 200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_metrics(epsilonrange, avgErr, avgSize, score_fn, save_plots_flag = True, res_path = None, show = False):
    # Plot with subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))


    axs[0].plot(epsilonrange, avgErr, 'b', linewidth=2, label='avgErr')
    axs[0].plot(epsilonrange, epsilonrange, 'r--', linewidth=2, label='Expected Error')
    axs[0].set_xlabel(r'$\varepsilon$', fontsize=16)
    axs[0].set_ylabel("Average Error", fontsize=16)
    axs[0].legend(fontsize=14)
    axs[0].grid(True)


    #axs[1].plot(epsilonrange, empty*100, linewidth=2, label='Empty')
    #axs[1].plot(epsilonrange, singleton*100, linewidth=2, label='Single')
    #axs[1].plot(epsilonrange, double*100, linewidth=2, label='Double')
    axs[1].plot(epsilonrange, avgSize, linewidth=2, label='AvgSize')
    axs[1].set_xlabel(r'$\varepsilon$', fontsize=16)
    #axs[1].set_ylabel("%", fontsize=16)
    axs[1].legend(fontsize=14)
    axs[1].grid(True)

    plt.tight_layout()
    if save_plots_flag and res_path!=None:
        plt.savefig(res_path+f"plots/CP_metrics_{score_fn}.png", dpi = 200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_calibration_scores_distribution(selectedscores_cal, score_fn, epsilon_vals = [0.01, 0.05, 0.1, 0.2], save_plots_flag = True, res_path = None, show = False):
    n_c = len(selectedscores_cal)
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    scores_calib_sorted = np.sort(selectedscores_cal)

    colors = ["red", "darkorange", "green", "violet"]
    ax.hist(selectedscores_cal, color = "lightskyblue", bins = 100)
    for i, epsilon in enumerate(epsilon_vals):
        q_level = np.ceil((n_c + 1) * (1 - epsilon))/n_c
        s_epsilon = np.quantile(scores_calib_sorted, q_level)
        ax.axvline(x = s_epsilon, linestyle = "--", color=colors[i], label = fr"$\varepsilon={epsilon}$")
        ax.set_xlabel("Score Values", fontsize = 16)
        ax.set_ylabel("Count", fontsize = 16)
        ax.grid(True)
    ax.legend()
    if save_plots_flag and res_path!=None:
        plt.savefig(res_path+f"plots/CalibScoresDistribution_{score_fn}.png", dpi = 200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_prediction_regions(Xts, Ycal, tau0cal, tau1cal, tau0ts, tau1ts, rule_limits, changeclsidx, wrong_0_ts, wrong_1_ts, score_fn, selected_eps = 0.01, save_plots_flag = True, res_path = None, show = False):
    n_c = len(Ycal)
    C_label, _, _, _ = GetPredictionRegions(Ycal, tau0cal, tau1cal, tau0ts, tau1ts, selected_eps, n_c)

    df = pd.DataFrame(np.hstack((Xts,C_label.reshape(-1,1))), columns = ["X1", "X2", "C"])

    fig, ax = plt.subplots(1,1,figsize=(7, 6))

    for r in range(rule_limits.shape[0]):

        if r < changeclsidx-1:
            ax.add_patch(plt.Rectangle((rule_limits[r,0], rule_limits[r,2]), abs(rule_limits[r,1]-rule_limits[r,0]), abs(rule_limits[r,3]-rule_limits[r,2]), linewidth=2, edgecolor='tab:green', linestyle='--', facecolor='none'))
        else:
            ax.add_patch(plt.Rectangle((rule_limits[r,0], rule_limits[r,2]), abs(rule_limits[r,1]-rule_limits[r,0]), abs(rule_limits[r,3]-rule_limits[r,2]), linewidth=2, edgecolor='tab:orange', linestyle='--', facecolor='none'))
            

    ax.scatter(Xts[C_label == 0, 0], Xts[C_label==0, 1], c="tab:green", s=100, edgecolors='black', linewidth=1.5, marker='o', label = "{0}")

    ax.scatter(Xts[C_label == 1, 0], Xts[C_label==1, 1], c="sandybrown", s=100, edgecolors='black', linewidth=1.5, marker='o', label = "{1}")

    ax.scatter(Xts[C_label == 2, 0], Xts[C_label==2, 1], c="yellow", s=100, edgecolors='black', linewidth=1.5, marker='o', label = "{0,1}")

    ax.scatter(Xts[C_label == 3, 0], Xts[C_label==3, 1], c="grey", s=100, edgecolors='black', linewidth=1.5, marker='o', label = "{}")

    ax.scatter(wrong_1_ts[:20, 0], wrong_1_ts[:20, 1], c='deepskyblue', s=30, marker='v', linewidth=2)
    ax.scatter(wrong_0_ts[:20, 0], wrong_0_ts[:20, 1], c='red', s=20, marker='^', linewidth=2)

    ax.legend(fontsize = 16)
    ax.set_xlabel("X1", fontsize=18)
    ax.set_ylabel("X2", fontsize=18)
    ax.tick_params(labelsize=16)
    if save_plots_flag and res_path!=None:
        plt.savefig(res_path+f"plots/conf_sets_{score_fn}_{selected_eps}.png", dpi = 200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

'''
# ## Plot of singleton sets errors

# 
fig, ax = plt.subplots(1,1, figsize = (10,6))
for score in ["margin", "knn", "lac", "confiderai+"]:
    ax.plot(epsilonrange, list(e1[score].values()), marker = 'o', linewidth = 2, label = score)
ax.set_xlabel(r"$\varepsilon$", fontsize = 16)
ax.set_ylabel("Error on singleton sets", fontsize = 16)
plt.legend(fontsize = 14)
if save_plots_flag:
    plt.savefig(res_path+f"plots/avgErr_singleton.png", dpi = 200, bbox_inches="tight")
# plt.show())

# 
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

fig = go.Figure()
d3_colors = px.colors.qualitative.D3 

for i, score in enumerate(["margin", "knn", "lac", "confiderai+"]):
    e1list = [e1[score][k] for k in epsilonrange]
    p2list = [p2[score][k] for k in epsilonrange]
    epslist = list(epsilonrange)

    fig.add_trace(
        go.Scatter3d(
            x=e1list,
            y=epslist,
            z=p2list,
            mode="lines+markers",
            name=score,
            line=dict(width=4, color=d3_colors[i % len(d3_colors)]),
            marker=dict(size=5, color=d3_colors[i % len(d3_colors)]),
            hovertemplate=(
                "e1: %{x:.3f}<br>"
                "ε: %{z:.2f}<extra></extra>"
                "p2: %{y:.3f}<br>"
            )
        )
    )

fig.update_layout(
    scene=dict(
        xaxis = dict(backgroundcolor="white", gridcolor="lightgray"),
        yaxis = dict(backgroundcolor="white", gridcolor="lightgray"),
        zaxis = dict(backgroundcolor="white", gridcolor="lightgray"),
        xaxis_title=dict(text="avgErr_Singleton"),
        yaxis_title=dict(text="ε"),
        zaxis_title=dict(text="avgDouble"),
    ),
    margin=dict(l=0, r=90, t=90, b=90),
    legend=dict(font_size=14),
    width=750,
    height=750
)


if save_plots_flag:
    pio.write_image(fig, f"{res_path}/plots/3d_ErrSingleton_size.png", width=750, height=750, scale = 2)
#fig.show()
'''