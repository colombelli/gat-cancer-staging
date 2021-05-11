import pandas as pd
from stellargraph import StellarGraph
from sklearn import preprocessing


target_encoding = preprocessing.LabelBinarizer()

def load_all_data(edges_file, features_file, classes_file):
    df_patients = pd.read_csv(edges_file)
    df_features = pd.read_csv(features_file, index_col=0)
    df_classes = pd.read_csv(classes_file, index_col=0).sample(frac=1)

    global target_encoding
    target_encoding.fit_transform(df_classes['class'])

    G = StellarGraph(edges=df_patients, nodes=df_features)
    return df_patients, df_features, df_classes, G


def binarize_data(data):
    return target_encoding.transform(data)