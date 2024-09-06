import numpy as np


FEATURES = 'examples'
LABEL = 'label_names'
TARGET = 'labels'
PREDICTION = 'predictions'
SET = 'sets'
SET_LABEL = 'set_labels'


def sample_and_split_dataset(ds, nb_train=1000, nb_calib=1000, nb_test=1000):
    np.random.seed(10)

    ds['train'] = ds['train'].class_encode_column(TARGET)
    df_train = ds['train'].train_test_split(
        train_size=nb_train, test_size=20, stratify_by_column=TARGET
    )['train']

    ds['eval'] = ds['eval'].class_encode_column(TARGET)
    df_calib = ds['eval'].train_test_split(
        train_size=nb_calib, test_size=20, stratify_by_column=TARGET
    )['train']

    ds['test'] = ds['test'].class_encode_column(TARGET)
    df_test = ds['test'].train_test_split(
        train_size=nb_test, test_size=20, stratify_by_column=TARGET
    )['train']
    
    return df_train, df_calib, df_test


def get_dataframe(*ds_list):
    for ds in ds_list:
        #df = ds.to_pandas()
        yield ds


def get_X_y(*df_list):
    for df in df_list:
        X, y = df[FEATURES], df[TARGET]
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        yield X, y


def get_label_mapping(df):
    df_indexes, df_labels = df[TARGET], df[LABEL]
    idx2lab = dict((lab, nam) for lab, nam in zip(df_indexes, df_labels))
    lab2idx = {val: key for key, val in idx2lab.items()}
    label_list = sorted(lab2idx, key=lambda nam: lab2idx[nam])
    return idx2lab, lab2idx, label_list
