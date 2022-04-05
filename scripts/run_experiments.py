from cProfile import label
from pickle import FALSE
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
from keras.callbacks import EarlyStopping
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(40)

from sklearn.model_selection import train_test_split
from stellargraph.mapper import FullBatchNodeGenerator

from data import DataManager
from models import get_gat_model, get_mlp_model
from losses import categorical_focal_loss


##################################################
################ INPUT PARAMETERS ################
##################################################

models_names = ["gat", "mlp"]  # In the order they are trained/evaluated

neurons_each_layer = [128, 64]
activations = ['elu', 'elu']
output_activation='softmax'
dropout=0.15
learning_rate=0.0005

#loss_function = 'categorical_crossentropy'
loss_functions = {
  "COAD": categorical_focal_loss(alpha=[1.6023, 0.6295, 0.8294, 1.7195], gamma=0),
  "KIRC": categorical_focal_loss(alpha=[0.5148, 2.5242, 1.0868, 1.3491], gamma=0),
  "LUAD": categorical_focal_loss(alpha=[0.4559, 1.0206, 1.5239, 5.8552], gamma=0)
}
classes = ["stage1", "stage2", "stage3", "stage4"]

attention_heads=8
attention_dropout=0

mlp_batch_size=8


# Build base paths
base_paths = []
b = "C:/Users/colombelli/Desktop/TCC/experiments/"
thresholds = {
  "COAD": [0.0025, 0.0030, 0.0035, 0.0040, 0.0045],
  "KIRC": [0.0025, 0.0030, 0.0035, 0.0040, 0.0045],
  "LUAD": [0.0015, 0.0020, 0.0025, 0.0030, 0.0035]
}
for cancer_type in ["KIRC", "COAD", "LUAD"]:
  ths = thresholds[cancer_type]
  for th in [str(t).replace('.', '') for t in ths]:
    base_paths.append((cancer_type, f"{b}{cancer_type}/snf/{th}/"))


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
training_epochs = 300
repetitions = 10
experiments_seed = 42

##################################################
##################################################
##################################################



from numpy.random import seed
seed(experiments_seed)

from tensorflow import random
random.set_seed(experiments_seed)



if __name__ =="__main__":
    

  for cancer_type, base_path in base_paths:
    seed(experiments_seed)
    random.set_seed(experiments_seed)

    print("\n\n##############################################################")
    print("Starting experiments for path:\n", base_path)
    print("##############################################################\n")

    dm = DataManager(base_path, models_names, classes)
    df_patients, df_features, df_classes, G = dm.load_all_data(only_cancer=True)


    for i in range(repetitions):

      X_train, X_test, y_train, y_test = train_test_split(df_features, 
                                          df_classes, stratify=df_classes, 
                                          test_size=0.2)

      X_validation, X_test, y_validation, y_test = train_test_split(X_test, 
                                                    y_test,
                                                    stratify=y_test, 
                                                    test_size=0.5)


      print(f"\nIteration: {i+1}")

      y_train = dm.binarize_data(y_train)
      y_validation = dm.binarize_data(y_validation)
      y_test = dm.binarize_data(y_test)

      # GAT generators
      generator = FullBatchNodeGenerator(G, method="gat")
      train_gen = generator.flow(X_train.index, y_train)
      validation_gen = generator.flow(X_validation.index, y_validation)
      test_gen = generator.flow(X_test.index, y_test)

      gat_model = get_gat_model(generator, y_train.shape[1], neurons_each_layer,
        activations, output_activation, attention_heads, dropout, 
        attention_dropout, learning_rate, loss_functions[cancer_type])

      mlp_model = get_mlp_model(y_train.shape[1], X_train.values.shape[1], 
        neurons_each_layer, dropout, activations, output_activation,
        learning_rate, loss_functions[cancer_type])
      
      print("\nTraining GAT model... ")
      gat_history = gat_model.fit(train_gen, epochs=training_epochs, 
          validation_data=(validation_gen), 
          callbacks=[early_stop], verbose=0, 
          shuffle=False,  # This should be False, since shuffling 
                          # data means shuffling the whole graph
      )

      gat_performance = gat_model.evaluate(test_gen)
      dm.write_to_results_csv(models_names[0], gat_performance)
      gat_pred = gat_model.predict(test_gen)[0]
      
      print("\nTraining MLP model...")
      mlp_history = mlp_model.fit(X_train, y_train, epochs=training_epochs, 
              validation_data=(X_validation, y_validation), 
              callbacks=[early_stop], batch_size=mlp_batch_size, verbose=0)
      mlp_performance = mlp_model.evaluate(X_test, y_test)
      dm.write_to_results_csv(models_names[1], mlp_performance)
      mlp_pred = mlp_model.predict(X_test)


      print("\nPlotting models' history (loss, acc, auc_roc, pr_auc) ...")
      dm.save_plots_history([gat_history, mlp_history], i)
      
      print("\nSaving confusion matrices...\n")
      dm.save_conf_matrices([gat_pred, mlp_pred], y_test, i)
