import pandas as pd
import numpy as np
import os

from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.model_selection import StratifiedKFold

from data import *
from models import *


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


base_path = "/home/colombelli/Documents/datasets/acgt/kidney/stellargraph/"
edges_file = base_path+"patients_edges.csv"
features_file = base_path+"patients_features.csv"
classes_file = base_path+"patients_classes.csv"

epochs = 500
k = 10
cross_validation_repetitions = 100
seed = 22

##################################################
##################################################
##################################################



if __name__ =="__main__":
    
    df_patients, df_features, df_classes, G = load_all_data(edges_file, features_file, classes_file)

    #for i in range(cross_validation_repetitions):
        # k-fold cross validation
     

