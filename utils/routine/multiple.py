from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.routine.single import get_prediction_on_trial


def run_fct(**kwargs):
    res = get_prediction_on_trial(**kwargs)
    return res[0], res[1], res[2], res[3][..., 0]


def get_prediction_on_several_trials(
    kwargs_experiment, num_splits=10,
    group_fct=None, label_fct=None, metric_fcts=None,
    exp_name='default'
):
    outputs = Parallel(n_jobs=-1)(
        delayed(run_fct)(**kwargs_experiment) for _ in range(num_splits)
    )
    X_compil, y_true_compil, y_pred_compil, y_pred_set_compil = map(np.array, zip(*outputs))

    X_compil = np.stack(X_compil, axis=0)
    y_true_compil = np.stack(y_true_compil, axis=0)
    y_pred_compil = np.stack(y_pred_compil, axis=0)
    y_pred_set_compil = np.stack(y_pred_set_compil, axis=0)

    df_all_results = pd.DataFrame()

    group_indexes = group_fct(
        X=X_compil, y_true_=y_true_compil,
        y_pred_=y_pred_compil, y_pred_set_=y_pred_set_compil,
    ) 

    for group in range(group_indexes.shape[-1]):
        indexes = group_indexes[..., group]
        dict_mixted = {
            'group': label_fct(group),
            'nb_samples': np.mean(group_indexes[..., group]),
            'num_splits': num_splits,
            'exp_name': exp_name,
        }
        for name, metric_fct in metric_fcts.items():
            dict_mixted[name] = metric_fct(
                X=X_compil, y_true_=y_true_compil,
                y_pred_=y_pred_compil, y_pred_set_=y_pred_set_compil,
                mask=indexes
            )
            
        for key, val in kwargs_experiment.items():
            if key in ['dataset']:
                continue
            elif callable(val):
                dict_mixted[key] = val.__name__
            elif isinstance(val, dict):
                for key_sub, val_sub in val.items():
                    key_name = f'{key}:{key_sub}'
                    if hasattr(val_sub, '__class__'):
                        dict_mixted[key_name] = val_sub.__class__.__name__  # TODO
                    else:
                        dict_mixted[key_name] = str(val_sub)
            else:
                dict_mixted[key] = str(val)

        df_cur = pd.DataFrame(dict_mixted)
        df_all_results = pd.concat([df_all_results, df_cur], ignore_index=True)
    
    return df_all_results


def plot_metrics(metrics, metric_names, figsize=None, **kwargs):
    nrows, ncols = 2, 2
    if figsize:
        figsize = (figsize[0], figsize[1] * nrows)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False, layout='constrained')

    kwargs.update({})

    for i, metric_name in enumerate(metric_names):
        sns.boxplot(
            ax=ax[i//nrows, i%ncols], data=metrics,
            x='group', y=metric_name, palette=sns.color_palette("pastel"), **kwargs
        ).set(
            title=metric_name + ' analysis',
            xlabel='group',
            ylabel=metric_name
    )
    
    # sns.boxplot(
    #     ax=ax[0,1], data=metrics,
    #     x='group', y='set_size', palette=sns.color_palette("pastel"), **kwargs
    # ).set(
    #     title='set_size' + ' analysis',
    #     xlabel='group',
    #     ylabel='set_size'
    # )
    
    # sns.boxplot(
    #     ax=ax[1,0], data=metrics,
    #     x='group', y='accuracy', palette=sns.color_palette("pastel"), **kwargs
    # ).set(
    #     title='accuracy' + ' analysis',
    #     xlabel='group',
    #     ylabel='accuracy'
    # )
    
    # sns.boxplot(
    #     ax=ax[1,1], data=metrics,
    #     x='group', y='nb_samples', palette=sns.color_palette("pastel"), **kwargs
    # ).set(
    #     title='nb_samples' + ' analysis',
    #     xlabel='group',
    #     ylabel='nb_samples'
    # )

    plt.show()
