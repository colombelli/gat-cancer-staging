import pandas as pd
from stellargraph import StellarGraph
from sklearn import preprocessing


target_encoding = preprocessing.LabelBinarizer()

def load_all_data(edges_file, features_file, classes_file, only_cancer=False):
    df_patients = pd.read_csv(edges_file)
    df_features = pd.read_csv(features_file, index_col=0)
    df_classes = pd.read_csv(classes_file, index_col=0).sample(frac=1)

    if only_cancer:
        cancer_samples = df_classes.loc[df_classes["class"] != "normal"].index
        df_classes = df_classes.loc[cancer_samples, :]
        df_features = df_features.loc[cancer_samples, :]
        df_patients = df_patients[df_patients['source'].isin(cancer_samples) & 
                                  df_patients['target'].isin(cancer_samples)]

    global target_encoding
    target_encoding.fit_transform(df_classes['class'])

    G = StellarGraph(edges=df_patients, nodes=df_features)
    return df_patients, df_features, df_classes, G


def binarize_data(data):
    return target_encoding.transform(data)