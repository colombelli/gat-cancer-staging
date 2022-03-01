from cProfile import label
from pickle import FALSE
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(40)

from sklearn.model_selection import StratifiedKFold
from stellargraph.mapper import FullBatchNodeGenerator

from data import DataManager
from models import get_gat_model, get_mlp_model
from losses import categorical_focal_loss


##################################################
################ INPUT PARAMETERS ################
##################################################

models_names = ["gat", "mlp"]  # In the order they are trained/evaluated

first_layer_size=32
second_layer_size=32
first_activation='elu'
second_activation='elu'
output_activation='softmax'
dropout=0.15
learning_rate=0.001

#loss_function='categorical_crossentropy'
#loss_function = categorical_focal_loss(alpha=[2.3276, 1.6831, 0.3852, 1.1163, 
#                                    2.0641]) # These weights were pre-computed
#loss_function = categorical_focal_loss(alpha=[1.5243, 0.4411, 0.9816, 17.0454])
#classes = ["stage1", "stage2", "stage3", "stage4"]
#loss_function = [categorical_focal_loss(alpha=[[1.5243, 0.3043, 17.0454]], gamma=0)]
#classes = ["stage1", "stage23", "stage4"]
loss_function = [categorical_focal_loss(alpha=[[1,1,1]], gamma=0)]
classes = [0,1,2]

attention_heads=8
attention_dropout=0.15

mlp_batch_size=8

base_paths = [
        "C:/Users/colombelli/Desktop/TCC/data/PMLB/waveform_40/r08/"
        #"C:/Users/colombelli/Desktop/TCC/data/TCGA/BRCA/06_005_100feat_stage23_gat_one_layer/"
        ]


training_epochs = 500
k = 10
cross_validation_repetitions = 1
experiments_seed = 42

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

    dm = DataManager(base_path, models_names, classes)
    df_patients, df_features, df_classes, G = dm.load_all_data(only_cancer=True)


    for i in range(cross_validation_repetitions):

      kfold = StratifiedKFold(n_splits=k, shuffle=True)
      j=0
      for train, test in kfold.split(df_features, df_classes):
        j+=1
        print(f"\nOuter iteration: {i+1} | k-Fold iteration: {j} | \
              Total: {(i+1)*j}")

        X_train = df_features.iloc[train]
        y_train = dm.binarize_data(df_classes.iloc[train])

        X_test = df_features.iloc[test]
        y_test = dm.binarize_data(df_classes.iloc[test])

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
        gat_history = gat_model.fit(train_gen, epochs=training_epochs, 
            verbose=0, shuffle=False,  # This should be False, since shuffling 
                                       # data means shuffling the whole graph
        )

        test_gen = generator.flow(X_test.index, y_test)
        gat_performance = gat_model.evaluate(test_gen)
        dm.write_to_results_csv(models_names[0], gat_performance)
        gat_pred = gat_model.predict(test_gen)[0]
        
        print("\nTraining MLP model...")
        mlp_history = mlp_model.fit(X_train, y_train, epochs=training_epochs, 
                      batch_size=mlp_batch_size, verbose=0)
        mlp_performance = mlp_model.evaluate(X_test, y_test)
        dm.write_to_results_csv(models_names[1], mlp_performance)
        mlp_pred = mlp_model.predict(X_test)


        print("\nPlotting models' history (loss, acc, auc_roc, pr_auc) ...")
        dm.save_plots_history([gat_history, mlp_history], i, j)
        
        print("\nSaving confusion matrices...\n")
        dm.save_conf_matrices([gat_pred, mlp_pred], y_test, i, j)
