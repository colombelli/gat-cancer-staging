import pandas as pd
from stellargraph import StellarGraph
from sklearn import preprocessing

base_path = "/home/colombelli/Documents/datasets/acgt/kidney/stellargraph/"
edges_file = base_path+"patients_edges.csv"
features_file = base_path+"patients_features.csv"
classes_file = base_path+"patients_classes.csv"

target_encoding = preprocessing.LabelBinarizer()

def load_all_data():
    df_patients = pd.read_csv(edges_file)
    df_features = pd.read_csv(features_file, index_col=0)
    df_classes = pd.read_csv(classes_file, index_col=0).sample(frac=1)

    global target_encoding
    target_encoding.fit_transform(df_classes['class'])

    G = StellarGraph(edges=df_patients, nodes=df_features)
    print(G.info())
    return df_patients, df_features, df_classes, G


def binarizer_fit(data):
    global target_encoding
    target_encoding.fit_transform(data)