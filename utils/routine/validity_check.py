import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from utils.routine.single import cumulative_average, get_coverage_on_trial


def get_prediction_on_several_delta_and_calib(
    kwargs_experiment, delta_array, nb_calib_array,
    num_splits=100, metric_fct=None,
):
    all_coverage_dict = {delta: [] for delta in delta_array}
    final_coverage_dict = {delta: [] for delta in delta_array}

    # Run experiment
    for delta in delta_array:
        for nb_calib in nb_calib_array:
            coverages_list = []

            def run_fct(kwargs_experiment, delta, nb_calib):
                kwargs_experiment['delta'] = float(delta)
                kwargs_experiment['nb_calib'] = int(nb_calib)
                kwargs_experiment['metric_fct'] = metric_fct
                try:
                    return get_coverage_on_trial(**kwargs_experiment)[0]
                except:
                    return np.nan

            coverages_list = Parallel(n_jobs=-1)(
                delayed(run_fct)(kwargs_experiment, delta, nb_calib) for _ in range(num_splits)
            )

            coverages_list = np.array(coverages_list)
            all_coverage_dict[delta].append(coverages_list)
            final_coverage = cumulative_average(coverages_list)[-1]
            final_coverage_dict[delta].append(final_coverage)

    return final_coverage_dict, all_coverage_dict


def plot_empirical_coverage_convergence_over_ncalib(
    final_coverage_dict, all_coverage_dict,
    target_coverage_list, nb_calib_array, nb_valid,
    figsize=None
):
    # Theorical bounds and exact coverage to attempt
    def lower_bound_fct(delta):
        return delta * np.ones_like(nb_calib_array)

    def upper_bound_fct(delta):
        return delta + 1/(nb_calib_array)

    def exact_coverage_fct(delta):
        L = np.floor((nb_calib_array+1)*(1-delta))
        return 1-L/(nb_calib_array+1)

    # Plot the results
    nrows, ncols = 1, len(target_coverage_list)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False, layout='constrained')

    for i, delta in enumerate(target_coverage_list):
        # Compute the different bounds, target
        cov = final_coverage_dict[delta]
        ub = upper_bound_fct(delta)
        lb = lower_bound_fct(delta)
        exact_cov = exact_coverage_fct(delta)
        ub = np.clip(ub, a_min=0, a_max=1)
        lb = np.clip(lb, a_min=0, a_max=1)
        nc_min, nc_max = np.min(nb_calib_array), np.max(nb_calib_array)

        L = np.floor((nb_calib_array+1)*(1-delta))
        R = len(all_coverage_dict[delta][0])
        std = np.sqrt(L*(nb_calib_array+1-L)*(nb_calib_array+nb_valid+1))/np.sqrt(nb_valid*R*(nb_calib_array+1)**2*(nb_calib_array+2))

        # Plot the results
        ax[0, i].fill_between(nb_calib_array, y1=exact_cov-3*std, y2=exact_cov+3*std, alpha=0.1, color='g')
        ax[0, i].plot(nb_calib_array, cov, alpha=0.5, color='g')
        ax[0, i].plot(nb_calib_array, lb, color='k', label='Lower Bound')
        ax[0, i].plot(nb_calib_array, ub, color='b', label='Upper Bound')
        ax[0, i].plot(nb_calib_array, exact_cov, color='g', ls='--', label='Exact Cov')
        ax[0, i].hlines(delta, nc_min, nc_max, color='r', ls='--', label='Target Cov')

        ax[0, i].legend(loc="upper right", ncol=2)
        ax[0, i].set_ylim(np.min(lb) - 0.05, 1.0)
        ax[0, i].set_xlabel(r'$nb_{calib}$')
        ax[0, i].set_ylabel(r'$\overline{\mathbb{C}}$')

    fig.suptitle(r'$\delta = $' + str(delta))
    plt.show()

# distribution of coverage
# distribution of empirical coverage with a finite validation set
# distribution of average empirical coverage
