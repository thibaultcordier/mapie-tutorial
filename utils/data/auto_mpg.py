import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from datasets import DatasetDict
from datasets import ClassLabel, Value


# mpg: continuous
# cylinders: multi-valued discrete
# displacement: continuous
# horsepower: continuous
# weight: continuous
# acceleration: continuous
# model year: multi-valued discrete
# origin: multi-valued discrete
# car name: string (unique for each instance)

FEATURES = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin'] #, 'car name']
LABEL = 'mpg'
TARGET = 'target'
PREDICTION = 'prediction'
SET = 'set'
SET_LABEL = 'set_label'


def sample_and_split_dataset(ds, nb_train=None, nb_calib=None, nb_test=None):
    # new_features = ds['train'].features.copy()
    # new_features["cylinders"] = ClassLabel(new_features["cylinders"])#.class_encode_column("cylinders")
    # new_features["model year"] = ClassLabel(new_features["model year"])#.class_encode_column("model year")
    # new_features["origin"] = ClassLabel(new_features["origin"])#.class_encode_column("origin")
    # new_features["car name"] = new_features["car name"].class_encode_column("car name")
    # ds['train'] = ds['train'].cast(new_features)
    
    if "car name" in ds['train'].column_names:
        ds['train'] = ds['train'].remove_columns(["car name"])
    ds['train'] = ds['train'].filter(
        lambda example: example["horsepower"] != "?"
    )

    if TARGET not in ds['train'].column_names:
        ds['train'] = ds['train'].add_column(
            name=TARGET, column=ds['train'][LABEL]
        )

    ds_tt = ds['train'].train_test_split(
        test_size=nb_calib+nb_test, train_size=nb_train,
    )
    ds_cv = ds_tt['test'].train_test_split(
        test_size=nb_test, train_size=nb_calib,
    )

    ds_tcv = DatasetDict({
        'train': ds_tt['train'],
        'calib': ds_cv['train'],
        'test': ds_cv['test']
    })

    return ds_tcv['train'], ds_tcv['calib'], ds_tcv['test']


def get_dataframe(*ds_list):
    for ds in ds_list:
        df = ds.to_pandas()
        yield df


def get_X_y(*df_list):
    for df in df_list:
        X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
        yield X, y


def get_label_mapping(ds):
    df_indexes, df_labels = ds['train'][TARGET], ds['train'][LABEL]
    idx2lab = dict((lab, nam) for lab, nam in zip(df_indexes, df_labels))
    lab2idx = {val: key for key, val in idx2lab.items()}
    label_list = sorted(lab2idx, key=lambda nam: lab2idx[nam])
    return idx2lab, lab2idx, label_list


def transform_set_to_label(arr):
    convert_array = lambda x: str(np.round(x[1]-x[0], 2))
    return np.apply_along_axis(convert_array, axis=1, arr=arr) 


def plot_data(df, figsize=None):
    #fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    sns.pairplot(data=df, height=1.)
    #.set(title=f'Auto MPG dataset (pairs of features)',)
    plt.show()


def get_yerr(y_pred, y_pis):
    return y_pred - y_pis[:, 0], y_pis[:, 1] - y_pred

def plot_predictions(df, conformalizer, alpha, figsize=None):
    y_test = df[TARGET].to_numpy()
    y_pred = df[PREDICTION].to_numpy()
    y_err = [df['error_down'].to_numpy(), df['error_up'].to_numpy()]

    in_indexes = np.logical_and(y_pred-y_err[0]<=y_test, y_test<=y_pred+y_err[1])
    out_indexes = np.logical_or(y_pred-y_err[0]>y_test, y_test>y_pred+y_err[1]) 

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=figsize, layout='constrained', sharey=True, squeeze=False)
    
    axs[0,0].scatter(
        y_test, y_pred, color="black"
    )
    axs[0,0].errorbar(
        y_test[in_indexes], y_pred[in_indexes], yerr=[y_err[0][in_indexes], y_err[1][in_indexes]],
        linestyle="None", color=sns.color_palette("pastel")[2], capsize=2
    )
    axs[0,0].errorbar(
        y_test[out_indexes], y_pred[out_indexes], yerr=[y_err[0][out_indexes], y_err[1][out_indexes]],
        linestyle="None", color=sns.color_palette("pastel")[1], capsize=2
    )
    axs[0,0].plot(
        [0, max(max(y_test), max(y_pred))], [0, max(max(y_test), max(y_pred))],
        "-", color=sns.color_palette("pastel")[0]
    )
    axs[0,0].set_xlabel("Actual label [MPG]")
    axs[0,0].set_ylabel("Predicted label [MPG]")
    axs[0,0].grid()
    #ax.set_title(f"{"test"} - coverage={cov:.0%}")
    plt.show()
