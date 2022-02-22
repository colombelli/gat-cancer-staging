import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv
from pathlib import Path
from stellargraph import StellarGraph
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

target_encoding = preprocessing.LabelBinarizer()


class DataManager:

    def __init__(self, base_path, models_names, classes) -> None:
        self.base_path = base_path
        self.models_names = models_names
        self.classes = classes
        self._create_results_directories()
        self._setup_results_csv()


    def _save_graph_info(self, G):
        with open(self.base_path+"graph_info.txt", "w") as f:
            print(G.info(), file=f)
        return

    def load_all_data(self, only_cancer=False):

        edges_file = self.base_path+"edges.csv"
        features_file = self.base_path+"features.csv"
        classes_file = self.base_path+"classes.csv"

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
        self._save_graph_info(G)
        return df_patients, df_features, df_classes, G


    def binarize_data(self, data):
        return target_encoding.transform(data)


    def write_to_results_csv(self, model_name, row):
        file_name = f"{self.base_path}{model_name}_results.csv"
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return


    def clear_plot(self):
        plt.figure().clear()
        plt.cla()
        plt.clf()
        plt.close('all')
        return


    def plot_save_metric(self, models_history, metric, i, j):
        fig = plt.figure()
        for i, history in enumerate(models_history):
            plt.plot(history.history[metric], label=self.models_names[i])
        plt.title(f'model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(self.models_names)
        fig.savefig(self.base_path + f"training_plots/{metric}/{i}_{j}.png")
        self.clear_plot()
        return


    def save_plots_history(self, models_history, i, j):
        self.plot_save_metric(models_history, 'loss', i, j)
        self.plot_save_metric(models_history, 'acc', i, j)
        self.plot_save_metric(models_history, 'auc_roc', i, j)
        self.plot_save_metric(models_history, 'auc_pr', i, j)
        return


    def save_conf_matrices(self, y_preds, y_test, i, j):

        y_true_classes = np.argmax(y_test, axis=1)
        for m, pred in enumerate(y_preds):
            pred = np.argmax(pred, axis=1)
            conf_matrix = confusion_matrix(y_true_classes, pred)
            df = pd.DataFrame(conf_matrix)
            df.index = self.classes
            df.columns = self.classes

            df.to_csv(f"{self.base_path}conf_matrices/"
                      f"{self.models_names[m]}_{i}_{j}.csv")
        return


    def _setup_results_csv(self):
        first_csv_row = ["loss", "acc", "auc_roc", "auc_pr"]
        for model_name in self.models_names:
            self.write_to_results_csv(model_name, first_csv_row)
        return


    def _create_results_directories(self):
        Path(self.base_path+"conf_matrices/").mkdir(parents=True, 
                                                    exist_ok=True)
        Path(self.base_path+"training_plots/").mkdir(parents=True, 
                                                    exist_ok=True)
        Path(self.base_path+"training_plots/loss/").mkdir(parents=True, 
                                                    exist_ok=True)
        Path(self.base_path+"training_plots/acc/").mkdir(parents=True, 
                                                    exist_ok=True)
        Path(self.base_path+"training_plots/auc_roc/").mkdir(parents=True,
                                                        exist_ok=True)
        Path(self.base_path+"training_plots/auc_pr/").mkdir(parents=True,
                                                        exist_ok=True)
        return
