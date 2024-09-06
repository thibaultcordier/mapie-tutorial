import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from datasets import DatasetDict


FEATURES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
LABEL = 'Species'
TARGET = 'target'
PREDICTION = 'prediction'
SET = 'set'
SET_LABEL = 'set_label'


def sample_and_split_dataset(ds, nb_train=None, nb_calib=None, nb_test=None):
    if TARGET not in ds['train'].column_names:
        ds['train'] = ds['train'].add_column(
            name=TARGET, column=ds['train'][LABEL]
        )
        ds['train'] = ds['train'].class_encode_column(TARGET)

    ds_tt = ds['train'].train_test_split(
        test_size=nb_calib+nb_test, train_size=nb_train,
        stratify_by_column=TARGET
    )
    ds_cv = ds_tt['test'].train_test_split(
        test_size=nb_test, train_size=nb_calib,
        stratify_by_column=TARGET
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


def get_set_mapping():
    set_list = ['001', '010', '100', '000', '011', '101', '110', '111']

    # set2color = {bin(i)[2:].zfill(3): code for i, code in enumerate(list(sns.color_palette("pastel").as_hex())[:2**3])}    
    set2color = {code: sns.color_palette("pastel").as_hex()[i] for i, code in enumerate(set_list)}
    set2color['000'] = '#eeeeee'

    return set2color, set_list


set2color, set_list = get_set_mapping()

def transform_set_to_label(arr):
    convert_array = lambda x: ''.join([str(int(elt)) for elt in x])
    return np.apply_along_axis(convert_array, axis=1, arr=arr) 


def plot_data(df, figsize=None):
    fig, ax = plt.subplots(
        ncols=4, figsize=figsize,
        layout='constrained'
    )
    feature_pairs = [
        ('SepalLengthCm', 'SepalWidthCm'), ('PetalLengthCm', 'PetalWidthCm'),
        ('SepalLengthCm', 'PetalLengthCm'), ('SepalWidthCm', 'PetalWidthCm')
    ]

    for i, (x, y) in enumerate(feature_pairs):
        sns.scatterplot(
            ax=ax[i], data=df, x=x, y=y, hue=LABEL, 
            legend=bool(i==len(feature_pairs)-1),
        ).set(
            xlabel=x, ylabel=y,
            title=f'Iris dataset\n({x} vs {y})',
        )
    sns.move_legend(ax[-1], loc='center left', bbox_to_anchor=(1., 0.5))
    plt.show()


def plot_predictions(df, conformalizer, alpha, nb_quantile=5, figsize=None):
    fig, ax = plt.subplots(
        ncols=nb_quantile, nrows=1, figsize=figsize,
        layout='constrained', sharey=True, squeeze=False
    )
    feature_pairs = [
        ('SepalLengthCm', 'SepalWidthCm'), ('PetalLengthCm', 'PetalWidthCm'),
        ('SepalLengthCm', 'PetalLengthCm'), ('SepalWidthCm', 'PetalWidthCm')
    ]

    for j in range(nb_quantile):
        for i, (x, y) in enumerate(feature_pairs[:1]):  # only plot first feature pair
            other_features = df[[f for f in FEATURES if f not in [x, y]]]
            mean_features = other_features.quantile(q=(j+1)/(nb_quantile+1))
            x_val = np.linspace(np.min(df[x]), np.max(df[x]), 50)
            y_val = np.linspace(np.min(df[y]), np.max(df[y]), 50)
            x_col, y_col = np.meshgrid(x_val, y_val)
            x_col, y_col = x_col.ravel(), y_col.ravel()
            new_df = pd.DataFrame({
                x: x_col, y: y_col,
                **{name: val*np.ones_like(x_col) for name, val in mean_features.items()}
            })
            _, y_pis = conformalizer.predict(new_df, alpha=alpha)
            new_df[PREDICTION] = transform_set_to_label(y_pis[:,:,0])
            sns.scatterplot(
                ax=ax[i, j], data=new_df, x=x, y=y, hue=PREDICTION,  # i,j reverted for sharey=True
                palette=set2color
            ).set(
                xlabel=x, ylabel=y,
                title=f'Iris dataset Prediction\n'
                + '\n'.join([name + ' = ' + str(round(val, 2)) for name, val in mean_features.items()]),
            )
    plt.show()
