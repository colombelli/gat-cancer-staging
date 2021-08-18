import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(40)

from sklearn.model_selection import StratifiedKFold
from stellargraph.mapper import FullBatchNodeGenerator

from data import load_all_data, binarize_data
from models import get_gat_model, get_mlp_model


##################################################
################ INPUT PARAMETERS ################
##################################################

first_layer_size=64
second_layer_size=64
first_activation='elu'
second_activation='elu'
output_activation='softmax'
dropout=0.15
learning_rate=0.0001
loss_function='categorical_crossentropy'

attention_heads=8
attention_dropout=0.15

mlp_batch_size=8

base_paths = [
        "/home/colombelli/Documents/Learning with graphs/gat_experiments/exp1/",
        ]


training_epochs = 10 # 500?
k = 10
cross_validation_repetitions = 5 ## 10?
experiments_seed = 22

##################################################
##################################################
##################################################



from numpy.random import seed
seed(experiments_seed)

from tensorflow import random
random.set_seed(experiments_seed)


if __name__ =="__main__":
    

    for base_path in base_paths:
        
        print("\n\n##############################################################")
        print("Starting experiments for path:\n", base_path)
        print("##############################################################\n")

        edges_file = base_path+"patients_edges.csv"
        features_file = base_path+"patients_features.csv"
        classes_file = base_path+"patients_classes.csv"

        df_patients, df_features, df_classes, G = load_all_data(edges_file, features_file, classes_file)
        with open(base_path+"graph_info.txt", "w") as f:
            print(G.info(), file=f)

        gat_evaluation = {"acc":[], "auc_roc":[], "auc_pr":[]}
        mlp_evaluation = {"acc":[], "auc_roc":[], "auc_pr":[]}
        for i in range(cross_validation_repetitions):

            kfold = StratifiedKFold(n_splits=k, shuffle=True)
            j=0
            for train, test in kfold.split(df_features, df_classes):
                j+=1
                print("\nOuter iteration: {} | k-Fold iteration: {} | Total: {}".format(i+1, j, (i+1)*j))



                X_train = df_features.iloc[train]
                y_train = binarize_data(df_classes.iloc[train])

                X_test = df_features.iloc[test]
                y_test = binarize_data(df_classes.iloc[test])

                # GAT generators
                generator = FullBatchNodeGenerator(G, method="gat")
                train_gen = generator.flow(X_train.index, y_train)


                gat_model = get_gat_model(generator, y_train.shape[1],
                  first_layer_size, second_layer_size, 
                  first_activation, second_activation, output_activation,
                  attention_heads, dropout, attention_dropout, 
                  learning_rate, loss_function)

                mlp_model = get_mlp_model(y_train.shape[1], X_train.values.shape[1], 
                  first_layer_size, second_layer_size, dropout,
                  first_activation, second_activation, output_activation,
                  learning_rate, loss_function)


                print("\nTraining GAT model... ")
                gat_model.fit(train_gen, epochs=training_epochs, verbose=0,
                    shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
                )

                test_gen = generator.flow(X_test.index, y_test)
                gat_performance = gat_model.evaluate(test_gen)

                gat_evaluation["acc"].append(gat_performance[1])
                gat_evaluation["auc_roc"].append(gat_performance[2])
                gat_evaluation["auc_pr"].append(gat_performance[3])
                

                print("\nTraining MLP model...")
                mlp_model.fit(X_train, y_train, epochs=training_epochs, batch_size=mlp_batch_size, verbose=0)
                mlp_performance = mlp_model.evaluate(X_test, y_test)

                mlp_evaluation["acc"].append(mlp_performance[1])
                mlp_evaluation["auc_roc"].append(mlp_performance[2])
                mlp_evaluation["auc_pr"].append(mlp_performance[3])
        

        pd.DataFrame.from_dict(gat_evaluation).to_csv(base_path+"gat_results.csv")
        pd.DataFrame.from_dict(mlp_evaluation).to_csv(base_path+"mlp_results.csv")
        