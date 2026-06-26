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
        plt.savefig(res_path+"plots/synthetic_classes.pdf", dpi = 200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def __plot_score(Xts, tau0ts, tau1ts, rule_limits, changeclsidx, wrong_1_ts, wrong_0_ts, save_plots_flag = True, score_fn = None, res_path = None, show = False):
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

    ax1.scatter(wrong_1_ts[:20, 0], wrong_1_ts[:20, 1], c='red', s=30, marker='x', linewidth=2)

    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('Score (class 0)', fontsize=16)
        #cbar.set_ticks([0, 1])
        #cbar.set_ticklabels(['0.0', '1.0'])
    ax1.set_xlabel("X1", fontsize=18)
    ax1.set_ylabel("X2", fontsize=18)
    ax1.tick_params(labelsize=16)


    sc = ax2.scatter(Xts[:, 0], Xts[:, 1], c= tau1ts, cmap='Blues',  norm = norm, s=120, edgecolors='black', linewidth=1.5, marker='o')
    ax2.scatter(wrong_0_ts[:20, 0], wrong_0_ts[:20, 1], c='red', s=30, marker='x', linewidth=2)


    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('Score (class 1)', fontsize=16)

    ax2.set_xlabel("X1", fontsize=18)
    ax2.set_ylabel("X2", fontsize=18)
    ax2.tick_params(labelsize=16)
    if save_plots_flag and res_path!=None:
        plt.savefig(res_path+f"plots/test_scores_{score_fn}.pdf", dpi = 200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_score(Xts, tau0ts, tau1ts, rule_limits, changeclsidx, wrong_1_ts, wrong_0_ts, save_plots_flag = True, score_fn = None, res_path = None, show = False):
    norm = colors.Normalize(vmin=0, vmax=1)

    # Plotting for tau0cal
    fig,ax1 = plt.subplots(1,1,figsize=(6, 6))
    

    for r in range(rule_limits.shape[0]):
        if r<changeclsidx-1:
            ax1.add_patch(plt.Rectangle((rule_limits[r,0], rule_limits[r,2]), abs(rule_limits[r,1]-rule_limits[r,0]), abs(rule_limits[r,3]-rule_limits[r,2]), linewidth=2, edgecolor='tab:green', linestyle='--', facecolor='none'))
        else:
            ax1.add_patch(plt.Rectangle((rule_limits[r,0], rule_limits[r,2]), abs(rule_limits[r,1]-rule_limits[r,0]), abs(rule_limits[r,3]-rule_limits[r,2]), linewidth=2, edgecolor='orange', linestyle='--', facecolor='none'))

    sc = ax1.scatter(Xts[:, 0], Xts[:, 1], c=tau0ts, cmap='Blues', norm = norm, s=120, edgecolors='black', linewidth=1.5, marker='o')

    ax1.scatter(wrong_1_ts[:20, 0], wrong_1_ts[:20, 1], c='red', s=30, marker='x', linewidth=2)

    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('Score (class 0)', fontsize=16)
        #cbar.set_ticks([0, 1])
        #cbar.set_ticklabels(['0.0', '1.0'])
    ax1.set_xlabel("X1", fontsize=18)
    ax1.set_ylabel("X2", fontsize=18)
    ax1.tick_params(labelsize=16)

    if save_plots_flag and res_path!=None:
        plt.savefig(res_path+f"plots/test_scores_class0_{score_fn}.pdf", dpi = 200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()   
    ####################################
    fig2, ax2 = plt.subplots(1,1,figsize=(6, 6))
    for r in range(rule_limits.shape[0]):
        if r<changeclsidx-1:
            ax2.add_patch(plt.Rectangle((rule_limits[r,0], rule_limits[r,2]), abs(rule_limits[r,1]-rule_limits[r,0]), abs(rule_limits[r,3]-rule_limits[r,2]), linewidth=2, edgecolor='tab:green', linestyle='--', facecolor='none'))
        else:
            ax2.add_patch(plt.Rectangle((rule_limits[r,0], rule_limits[r,2]), abs(rule_limits[r,1]-rule_limits[r,0]), abs(rule_limits[r,3]-rule_limits[r,2]), linewidth=2, edgecolor='orange',linestyle='--', facecolor='none'))

    sc = ax2.scatter(Xts[:, 0], Xts[:, 1], c= tau1ts, cmap='Blues',  norm = norm, s=120, edgecolors='black', linewidth=1.5, marker='o')
    ax2.scatter(wrong_0_ts[:20, 0], wrong_0_ts[:20, 1], c='red', s=30, marker='x', linewidth=2)


    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('Score (class 1)', fontsize=16)

    ax2.set_xlabel("X1", fontsize=18)
    ax2.set_ylabel("X2", fontsize=18)
    ax2.tick_params(labelsize=16)
    if save_plots_flag and res_path!=None:
        plt.savefig(res_path+f"plots/test_scores_class1_{score_fn}.pdf", dpi = 200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()    

def plot_score_rulewise(
        Xts_rules,
        scores0_test_knn,
        scores1_test_knn,
        rule_limits,
        changeclsidx,
        wrong_1_ts=None,
        wrong_0_ts=None,
        save_plots_flag=True,
        score_fn="knn",
        res_path=None,
        show=False):

    norm = colors.Normalize(vmin=0, vmax=1)

    ############################
    # CLASS 0
    ############################
    fig, ax = plt.subplots(figsize=(6, 6))

    # draw rules
    for r in range(rule_limits.shape[0]):

        color = 'tab:green' if r < changeclsidx-1 else 'orange'

        ax.add_patch(
            plt.Rectangle(
                (rule_limits[r,0], rule_limits[r,2]),
                abs(rule_limits[r,1]-rule_limits[r,0]),
                abs(rule_limits[r,3]-rule_limits[r,2]),
                linewidth=2,
                edgecolor=color,
                linestyle='--',
                facecolor='none'
            )
        )

    # plot each rule separately
    for r, Xr in enumerate(Xts_rules):

        if len(scores0_test_knn[r]) == 0:
            continue

        sc = ax.scatter(
            Xr[:,0],
            Xr[:,1],
            c=scores0_test_knn[r],
            cmap="Blues",
            norm=norm,
            s=120,
            edgecolors="black",
            linewidth=1.5
        )

    if wrong_1_ts is not None and len(wrong_1_ts) > 0:
        ax.scatter(
            wrong_1_ts[:,0],
            wrong_1_ts[:,1],
            c="red",
            marker="x",
            s=30,
            linewidth=2
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Score (class 0)", fontsize=16)

    ax.set_xlabel("X1", fontsize=18)
    ax.set_ylabel("X2", fontsize=18)
    ax.tick_params(labelsize=16)

    if save_plots_flag and res_path is not None:
        plt.savefig(
            f"{res_path}/plots/test_scores_class0_{score_fn}.pdf",
            dpi=200,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    plt.close()

    ############################
    # CLASS 1
    ############################
    fig, ax = plt.subplots(figsize=(6, 6))

    for r in range(rule_limits.shape[0]):

        color = 'tab:green' if r < changeclsidx-1 else 'orange'

        ax.add_patch(
            plt.Rectangle(
                (rule_limits[r,0], rule_limits[r,2]),
                abs(rule_limits[r,1]-rule_limits[r,0]),
                abs(rule_limits[r,3]-rule_limits[r,2]),
                linewidth=2,
                edgecolor=color,
                linestyle='--',
                facecolor='none'
            )
        )

    for r, Xr in enumerate(Xts_rules):

        if len(scores1_test_knn[r]) == 0:
            continue

        sc = ax.scatter(
            Xr[:,0],
            Xr[:,1],
            c=scores1_test_knn[r],
            cmap="Blues",
            norm=norm,
            s=120,
            edgecolors="black",
            linewidth=1.5
        )

    if wrong_0_ts is not None and len(wrong_0_ts) > 0:
        ax.scatter(
            wrong_0_ts[:,0],
            wrong_0_ts[:,1],
            c="red",
            marker="x",
            s=30,
            linewidth=2
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Score (class 1)", fontsize=16)

    ax.set_xlabel("X1", fontsize=18)
    ax.set_ylabel("X2", fontsize=18)
    ax.tick_params(labelsize=16)

    if save_plots_flag and res_path is not None:
        plt.savefig(
            f"{res_path}/plots/test_scores_class1_{score_fn}.pdf",
            dpi=200,
            bbox_inches="tight"
        )

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
        plt.savefig(res_path+f"plots/CP_metrics_{score_fn}.pdf", dpi = 200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_calibration_scores_distribution(selectedscores_cal, score_fn, epsilon_vals = [0.01, 0.05, 0.1, 0.2], save_plots_flag = True, res_path = None, show = False):
    n_c = len(selectedscores_cal)
    fig, ax = plt.subplots(1,1,figsize=(5,3))
    scores_calib_sorted = np.sort(selectedscores_cal)

    colors = ["red", "darkorange", "lime", "magenta"]
    ax.hist(selectedscores_cal, color = "royalblue", bins = 20)
    for i, epsilon in enumerate(epsilon_vals):
        q_level = np.ceil((n_c + 1) * (1 - epsilon))/n_c
        s_epsilon = np.quantile(scores_calib_sorted, q_level)
        ax.axvline(x = s_epsilon, linestyle = "--", color=colors[i],linewidth = 2, label = fr"$\varepsilon={epsilon}$")
        ax.set_xlabel("Score Values", fontsize = 18)
        ax.set_ylabel("Count", fontsize = 18)
        ax.grid(True)
    ax.legend(loc = "upper right", fontsize = 16)
    if save_plots_flag and res_path!=None:
        plt.savefig(res_path+f"plots/CalibScoresDistribution_{score_fn}.pdf", dpi = 200, bbox_inches="tight")
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
            

    ax.scatter(Xts[C_label == 0, 0], Xts[C_label==0, 1], c="blue", s=100, edgecolors='black', linewidth=1.5, marker='o', label = "{0}")

    ax.scatter(Xts[C_label == 1, 0], Xts[C_label==1, 1], c="red", s=100, edgecolors='black', linewidth=1.5, marker='o', label = "{1}")

    ax.scatter(Xts[C_label == 2, 0], Xts[C_label==2, 1], c="yellow", s=100, edgecolors='black', linewidth=1.5, marker='o', label = "{0,1}")

    ax.scatter(Xts[C_label == 3, 0], Xts[C_label==3, 1], c="grey", s=100, edgecolors='black', linewidth=1.5, marker='o', label = "{}")

    ax.scatter(wrong_1_ts[:20, 0], wrong_1_ts[:20, 1], c='deepskyblue', s=30, marker='x', linewidth=2)
    ax.scatter(wrong_0_ts[:20, 0], wrong_0_ts[:20, 1], c='violet', s=30, marker='x', linewidth=2)

    ax.legend(fontsize = 16)
    ax.set_xlabel("X1", fontsize=18)
    ax.set_ylabel("X2", fontsize=18)
    ax.tick_params(labelsize=16)
    if save_plots_flag and res_path!=None:
        plt.savefig(res_path+f"plots/conf_sets_{score_fn}_{selected_eps}.pdf", dpi = 200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_prediction_regions_knn(
        Xts_rules,
        C_all,
        rule_limits,
        changeclsidx,
        wrong_0_ts=None,
        wrong_1_ts=None,
        score_fn="knn",
        selected_eps=0.01,
        save_plots_flag=True,
        res_path=None,
        show=False):

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        # Disegna le regole
        for r in range(rule_limits.shape[0]):

            color = "tab:green" if r < changeclsidx-1 else "tab:orange"

            ax.add_patch(
                plt.Rectangle(
                    (rule_limits[r, 0], rule_limits[r, 2]),
                    abs(rule_limits[r, 1] - rule_limits[r, 0]),
                    abs(rule_limits[r, 3] - rule_limits[r, 2]),
                    linewidth=2,
                    edgecolor=color,
                    linestyle="--",
                    facecolor="none"
                )
            )


        first0 = True
        first1 = True
        first2 = True
        first3 = True

        for r in range(len(Xts_rules)):

            Xr = Xts_rules[r]

            if len(Xr) == 0:
                continue

            Cr = np.asarray(C_all[r])

            if np.any(Cr == 0):
                ax.scatter(
                    Xr[Cr == 0, 0],
                    Xr[Cr == 0, 1],
                    c="blue",
                    s=100,
                    edgecolors="black",
                    linewidth=1.5,
                    marker="o",
                    label="{0}" if first0 else None
                )
                first0 = False

            if np.any(Cr == 1):
                ax.scatter(
                    Xr[Cr == 1, 0],
                    Xr[Cr == 1, 1],
                    c="red",
                    s=100,
                    edgecolors="black",
                    linewidth=1.5,
                    marker="o",
                    label="{1}" if first1 else None
                )
                first1 = False

            if np.any(Cr == 2):
                ax.scatter(
                    Xr[Cr == 2, 0],
                    Xr[Cr == 2, 1],
                    c="yellow",
                    s=100,
                    edgecolors="black",
                    linewidth=1.5,
                    marker="o",
                    label="{0,1}" if first2 else None
                )
                first2 = False

            if np.any(Cr == 3):
                ax.scatter(
                    Xr[Cr == 3, 0],
                    Xr[Cr == 3, 1],
                    c="grey",
                    s=100,
                    edgecolors="black",
                    linewidth=1.5,
                    marker="o",
                    label="{}" if first3 else None
                )
                first3 = False


        if wrong_1_ts is not None and len(wrong_1_ts) > 0:
            ax.scatter(
                wrong_1_ts[:, 0],
                wrong_1_ts[:, 1],
                c="deepskyblue",
                s=30,
                marker="x",
                linewidth=2
            )

        if wrong_0_ts is not None and len(wrong_0_ts) > 0:
            ax.scatter(
                wrong_0_ts[:, 0],
                wrong_0_ts[:, 1],
                c="violet",
                s=30,
                marker="x",
                linewidth=2
            )

        ax.legend(fontsize=16)

        ax.set_xlabel("X1", fontsize=18)
        ax.set_ylabel("X2", fontsize=18)
        ax.tick_params(labelsize=16)

        if save_plots_flag and res_path is not None:
            plt.savefig(
                f"{res_path}/plots/conf_sets_{score_fn}_{selected_eps}.pdf",
                dpi=200,
                bbox_inches="tight"
            )

        if show:
            plt.show()

        plt.close()

