import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_prediction_on_trial(
    predictor, conformalizer,
    dataset, delta, nb_train, nb_calib, nb_test,
    sample_and_split_function=None, get_dataframe=None, get_X_y=None,
    kwargs_predictor=dict(), kwargs_conformalizer=dict(), kwargs_predict=dict(),
    random_state=None
):
    """
    """
    # 0) Split data step
    ds_train, ds_calib, ds_test = sample_and_split_function(dataset, nb_train, nb_calib, nb_test)
    df_train, df_calib, df_test = get_dataframe(ds_train, ds_calib, ds_test)
    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = get_X_y(df_train, df_calib, df_test)

    # 1) Get main predictor.
    predictor = predictor(**kwargs_predictor)
    predictor.fit(X_train, y_train)

    # 2) Conformalize the main predictor.
    conformalizer = conformalizer(estimator=predictor, **kwargs_conformalizer)
    conformalizer.fit(X_calib, y_calib)

    # 3) Obtain predictions and sets of predictions.
    y_pred, y_pis = conformalizer.predict(X_test, alpha=1-delta, **kwargs_predict)

    return X_test, y_test, y_pred, y_pis


def get_coverage_on_trial(
    predictor, conformalizer,
    dataset, delta, nb_train, nb_calib, nb_test,
    sample_and_split_function=None, get_dataframe=None, get_X_y=None,
    metric_fct=None,
    kwargs_predictor=dict(), kwargs_conformalizer=dict(), kwargs_predict=dict(),
    random_state=None
):
    """
    Calculate the fraction of test samples within the predicted intervals.

    This function splits the data into a training set and a test set. If the
    cross-validation strategy of the mapie regressor is a ShuffleSplit, it fits
    the regressor to the entire training set. Otherwise, it further splits the
    training set into a calibration set and a training set, and fits the
    regressor to the calibration set. It then predicts intervals for the test
    set and calculates the fraction of test samples within these intervals.

    Parameters:
    -----------
    predictor: object
        A regressor object.

    conformalizer: object
        A mapie regressor object.

    data: array-like of shape (n_samples, n_features)
        The data to be split into a training set and a test set.

    target: array-like of shape (n_samples,)
        The target values for the data.

    delta: float
        The level of confidence for the predicted intervals.

    Returns:
    --------
    fraction_within_bounds: float
        The fraction of test samples within the predicted intervals.
    """
    # 0) Split data step
    ds_train, ds_calib, ds_test = sample_and_split_function(dataset, nb_train, nb_calib, nb_test)
    df_train, df_calib, df_test = get_dataframe(ds_train, ds_calib, ds_test)
    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = get_X_y(df_train, df_calib, df_test)

    # 1) Get main predictor.
    predictor = predictor(**kwargs_predictor)
    predictor.fit(X_train, y_train)

    # 2) Conformalize the main predictor.
    conformalizer = conformalizer(estimator=predictor, **kwargs_conformalizer)
    conformalizer.fit(X_calib, y_calib)

    # 3) Obtain predictions and sets of predictions.
    y_pred, y_pis = conformalizer.predict(X_test, alpha=1-delta, **kwargs_predict)

    # 4) Coverage step
    coverage = metric_fct(y_test, y_pis)

    return coverage


def cumulative_average(arr):
    """
    Calculate the cumulative average of a list of numbers.

    This function computes the cumulative average of a list of numbers by
    calculating the cumulative sum of the numbers and dividing it by the
    index of the current number.

    Parameters:
    -----------
    arr: List[float]
        The input list of numbers.

    Returns:
    --------
    running_avg: List[float]
        The cumulative average of the input list.
    """
    cumsum = np.cumsum(arr)
    indices = np.arange(1, len(arr) + 1)
    cumulative_avg = cumsum / indices
    return cumulative_avg


def plot_empirical_coverage_convergence(
    empirical_coverages,
    target_coverage, nb_calib, nb_val,
    figsize=None, only_distribution=False
):
    ncols = 1 if only_distribution else 2
    fig, ax = plt.subplots(ncols=ncols, figsize=figsize, squeeze=False, layout='constrained')

    kwargs = {'bins': 30, 'binrange': [0, 1]}
    kwargs.update({'stat': 'proportion', 'kde': True})
    (
        sns.histplot(
            ax=ax[0, 0], data={'score': empirical_coverages}, x='score', **kwargs
        ).set(
            title='Distribution of coverage scores',
            xlabel='Coverage',
            ylabel='Frequencies'
        )
    )
    ax[0, 0].axvline(
        x=target_coverage,
        ymin=0, ymax=1, ls="dashed",
        label=f"target coverage = {target_coverage:.1f}"
    )
    avg_cov = np.mean(empirical_coverages)
    ax[0, 0].axvline(
        x=target_coverage,
        ymin=0, ymax=1, ls="dashed", color='red',
        label=f"average effective coverage = {avg_cov:.1f}"
    )
    
    if only_distribution:
        plt.tight_layout()
        plt.show()
        return
    
    # Compute theorical bounds and exact coverage to attempt
    delta = target_coverage
    num_splits = len(empirical_coverages)

    lower_bound = delta
    upper_bound = (delta + 1/(nb_calib+1))
    upper_bound_2 = (delta + 1/(nb_calib/2+1))
    exact_cov = (np.ceil((nb_calib+1)*delta))/(nb_calib+1)

    cumulative_averages_mapie = cumulative_average(empirical_coverages)

    # # Plot the results
    L = np.floor((nb_calib+1)*(1-delta))
    R = np.arange(0, num_splits)
    std = np.sqrt(L*(nb_calib+1-L)*(nb_calib+nb_val+1))/np.sqrt(nb_val*R*(nb_calib+1)**2*(nb_calib+2))
    ax[0, 1].fill_between(np.arange(0, num_splits), y1=exact_cov-3*std, y2=exact_cov+3*std, alpha=0.1, color='g')

    # Plot the results
    ax[0, 1].plot(cumulative_averages_mapie, alpha=0.5, label='MAPIE', color='g')

    ax[0, 1].hlines(exact_cov, 0, num_splits, color='r', ls='--', label='Exact Cov.')
    ax[0, 1].hlines(lower_bound, 0, num_splits, color='k', label='Lower Bound')
    ax[0, 1].hlines(upper_bound, 0, num_splits, color='b', label='Upper Bound')

    ax[0, 1].set_xlabel(r'Split Number')
    ax[0, 1].set_ylabel(r'$\overline{\mathbb{C}}$')
    ax[0, 1].set_title(r'$|D_{cal}| = $' + str(nb_calib) + r' and $\delta = $' + str(delta))

    ax[0, 1].legend(loc="upper right", ncol=2)
    ax[0, 1].set_ylim(0.7, 1)
    plt.tight_layout()
    plt.show()
