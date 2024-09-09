import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_conformity_score_distribution(
    scores, q, a, a_cor, figsize=None, kwargs_hist=dict(), kwargs_ecdf=dict()
):
    fig, ax = plt.subplots(ncols=2, figsize=figsize, layout='constrained')

    kwargs_hist.update({'stat': 'proportion', 'kde': True})
    sns.histplot(
        ax=ax[0], data={'score': scores}, x='score', **kwargs_hist
    ).set(
        title='Distribution of conformity scores (one trial)',
        xlabel='Conformity scores',
        ylabel='Frequencies',
        xlim=None if 'binrange' not in kwargs_hist else kwargs_hist['binrange']
    )

    kwargs_ecdf.update({'stat': 'proportion'})
    sns.ecdfplot(
        ax=ax[1], data={'score': scores}, x='score', **kwargs_ecdf
    ).set(
        title='CDF of conformity scores (one trial)',
        xlabel='Conformity scores',
        ylabel='Frequencies',
        xlim=None if 'binrange' not in kwargs_hist else kwargs_hist['binrange'],
    )
    ax[1].axhline(
        y=1-a, xmin=0, xmax=q,
        ls="dashed", color=sns.color_palette("pastel")[1],
        label=f"target coverage = (1 - α)"
    )
    ax[1].axhline(
        y=1-a_cor, xmin=0, xmax=q,
        ls="dashed", color=sns.color_palette("pastel")[2],
        label=r"corrected target = $\lceil(n + 1)(1 - α)\rceil /$ n"
    )
    ax[1].axvline(
        x=q, ymin=0, ymax=1-a,
        ls="dashed", color=sns.color_palette("pastel")[3],
        label=f"target-quantile"  # {q:.2f}
    )
    ax[0].axvline(
        x=q, ymin=0, ymax=1,
        ls="dashed", color=sns.color_palette("pastel")[3],
        label=f"target-quantile"  # {q:.2f}
    )
    ax[1].legend()
    #sns.move_legend(ax[1], loc='center left', bbox_to_anchor=(1., 0.5))
    plt.show()


def plot_size_set_distribution(size_sets, label=None, figsize=None, **kwargs):
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    if not isinstance(size_sets, pd.DataFrame):
        size_sets = pd.DataFrame({'size_set': size_sets})
    kwargs.update({'stat': 'proportion', 'common_norm': False})
    sns.histplot(
        ax=ax, data=size_sets, x='size_set', **kwargs
    ).set(
        title="Distribution of prediction set size",
        xlabel="Prediction set size",
        ylabel="Frequences"
    )
    # if label:
    #     size_sets = size_sets.sort_values(label)
    #     sns.histplot(data={'size_set': size_sets}, x='size_set', hue=label, multiple='dodge', **kwargs)
    try:
        sns.move_legend(ax, loc='center left', bbox_to_anchor=(1., 0.5))#, labels=label_list)
    except:
        ()
    
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


def plot_conformal_metric_distribution(
    target_coverages, effective_coverages, set_sizes,
    nb_calib, nb_val,
    figsize=None
):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, layout='constrained')

    for i, coverage in enumerate(effective_coverages):
        axes[0].plot(target_coverages, coverage)  #, label=method)
    axes[0].plot([0, 1], [0, 1], ls="--", color="k")
    axes[0].set_xlabel("Target coverage (1-alpha)")
    axes[0].set_ylabel("Effective coverage")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # # Plot confidence interval    
    # def exact_coverage_fct(delta):
    #     L = np.floor((nb_calib+1)*(1-delta))
    #     return 1-L/(nb_calib+1)

    # exact_cov = exact_coverage_fct(target_coverages)
    # L = np.floor((nb_calib+1)*target_coverages)
    # R = 1
    # std = np.sqrt(L*(nb_calib+1-L)*(nb_calib+nb_val+1))/np.sqrt(nb_val*R*(nb_calib+1)**2*(nb_calib+2))

    # axes[0].fill_between(target_coverages, y1=exact_cov-3*std, y2=exact_cov+3*std, alpha=0.1, color='g')

    for i, width in enumerate(set_sizes):
        axes[1].plot(target_coverages, width, label=set_sizes)
    axes[1].set_xlabel("Target coverage (1-alpha)")
    axes[1].set_ylabel("Average of prediction set sizes")
    axes[1].set_xlim(0, 1)

    plt.suptitle(
        "Effective coverage and prediction set size" #f" for the {method} method"
    )
    plt.show()


def plot_metrics_inside_groups(metrics_, labels_, number_obs_, figsize=None):
    x = np.arange(len(labels_))  # the label locations
    width = 1/(len(metrics_)+1)  # the width of the bars
    multiplier = 0

    fig, ax1 = plt.subplots(figsize=figsize, layout='constrained')

    # Define a refined color palette
    colors = sns.color_palette("pastel")

    for attribute, measurement in zip(metrics_.keys(), metrics_.values()):
        if attribute == 'set size':
            continue
        offset = width * multiplier
        rects = ax1.bar(x + offset, measurement, width, label=attribute, color=colors[multiplier])
        ax1.bar_label(rects, padding=3, fontsize=10, color='black')
        multiplier += 1

    # Create a secondary y-axis for number of observations
    ax2 = ax1.twinx()
    ax2.plot(
        x + offset/2, metrics_['set size'],
        color='dimgrey', marker='o', label='Set Size', linestyle='--'
    )
    ax2.set_ylabel('Set Size', color='dimgrey', fontsize=12)
    #ax2.set_ylabel('Number of Observations', color='dimgrey', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='dimgrey')
    ax2.set_ylim(0, max(metrics_['set size']) + 1)  # Adjust the y-axis limit as needed

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax1.set_title('Accuracy and Coverage Metrics by Set Size', fontsize=14)
    ax1.set_xlabel('CP Set Size', fontsize=12)
    ax1.set_ylabel('Metrics', fontsize=12)
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(labels_)
    ax1.legend(loc='best', ncols=2, fontsize=10)
    ax1.set_ylim(0, 1.1)  # Extend y-axis a bit for better label visibility

    # Add gridlines for primary and secondary y-axes
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # Show the plot
    fig.tight_layout()
    plt.show()
