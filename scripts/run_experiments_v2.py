import os

from stellargraph import StellarGraph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras_tuner as kt

tf.compat.v1.logging.set_verbosity(40)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from stellargraph.mapper import FullBatchNodeGenerator

from data import DataManager
from models import get_gat_model, get_mlp_model, get_gcn_model
from build_network import NearestNeighborsNet
from contextlib import redirect_stdout


##################################################
################ INPUT PARAMETERS ################
##################################################

models_names = ["gat", "gcn", "mlp"]  # In the order they are trained/evaluated

possible_num_layers = [1, 2, 3]
possible_num_neurons = [32, 64, 128]
activations_function = 'elu'
output_activation = 'softmax'
possible_dropouts = [0.0, 0.1, 0.2, 0.3]
possible_lrs = [0.0001, 0.0005, 0.001, 0.005]
possible_gammas = [0, 1, 2]
loss_function_weights = {
  "COAD": [1.6023, 0.6295, 0.8294, 1.7195],#categorical_focal_loss(alpha=[1.6023, 0.6295, 0.8294, 1.7195], gamma=0),
  "KIRC": [0.5148, 2.5242, 1.0868, 1.3491],#categorical_focal_loss(alpha=[0.5148, 2.5242, 1.0868, 1.3491], gamma=0),
  "LUAD": [0.4559, 1.0206, 1.5239, 5.8552]#categorical_focal_loss(alpha=[0.4559, 1.0206, 1.5239, 5.8552], gamma=0)
}
classes = ["stage1", "stage2", "stage3", "stage4"]

possible_attention_heads = [2, 4, 8]
attention_dropout = 0

mlp_batch_size=8


# Build base paths
base_paths = []
b = "C:/Users/colombelli/Desktop/TCC/experiments_knn/"
for cancer_type in ["KIRC", "COAD", "LUAD"]:
    for n_neighbors in range(2,11):
      base_paths.append((cancer_type, f"{b}{cancer_type}/{n_neighbors}/"))


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
training_epochs = 300
hp_epochs = 10
hp_early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
repetitions = 10
experiments_seed = 42
mlp_only_once = False
train_mlp = False

z_normalize_features = False
perform_feature_selection = False
num_features = 50

##################################################
##################################################
##################################################



from numpy.random import seed
seed(experiments_seed)

from tensorflow import random
random.set_seed(experiments_seed)


from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout



if __name__ =="__main__":
    
  prev_cancer_type = None
  for cancer_type, base_path in base_paths:
    seed(experiments_seed)
    random.set_seed(experiments_seed)

    print("\n\n##############################################################")
    print("Starting experiments for path:\n", base_path)
    print("##############################################################\n")

    dm = DataManager(base_path, models_names, classes)
    root_cancer_path = base_path + "../"
    df_features, df_classes = dm.load_classes_and_features(root_cancer_path,
                                                          only_cancer=True)

    n_neighbors = int(base_path.split("/")[-2])
    nnn = NearestNeighborsNet(df_features, df_classes, n_neighbors)
    edges = nnn.all_data_net()
    edges.to_csv(base_path+"edges.csv", index=False)
    G = StellarGraph(edges=edges, nodes=df_features)
    dm.save_graph_info(G)


    if mlp_only_once:
      if prev_cancer_type != cancer_type:
        prev_cancer_type = cancer_type
        train_mlp = True
      else:
        train_mlp = False


    for i in range(repetitions):

      print(f"\nIteration: {i+1}")

      X_train, X_test, y_train, y_test = train_test_split(df_features, 
                                          df_classes, stratify=df_classes, 
                                          test_size=0.2)

      X_validation, X_test, y_validation, y_test = train_test_split(X_test, 
                                                    y_test,
                                                    stratify=y_test, 
                                                    test_size=0.5)
      if perform_feature_selection:
        print("\nPerforming feature selection...")
        forest = RandomForestClassifier()
        forest.fit(X_train, y_train['class'])
        importances = forest.feature_importances_
        f_sorted = [f[0] for f in sorted(enumerate(importances),key=lambda i:i[1])]

        best_features = f_sorted[-num_features:]
        X_train = X_train.iloc[:, best_features]
        X_validation = X_validation.iloc[:, best_features]
        X_test = X_test.iloc[:, best_features]


      X_train_index = X_train.index
      X_validation_index = X_validation.index
      X_test_index = X_test.index
      
      if z_normalize_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_validation)
        X_test = scaler.transform(X_test)
      else:
        X_train = X_train.values
        X_validation = X_validation.values
        X_test = X_test.values
      
      histories = []
      preds = []

      y_train = dm.binarize_data(y_train)
      y_validation = dm.binarize_data(y_validation)
      y_test = dm.binarize_data(y_test)

      # GAT generators
      generator = FullBatchNodeGenerator(G, method="gat")
      train_gen = generator.flow(X_train_index, y_train)
      validation_gen = generator.flow(X_validation_index, y_validation)
      test_gen = generator.flow(X_test_index, y_test)

      gat_model_builder = get_gat_model(generator, y_train.shape[1], 
                    possible_num_layers, possible_gammas, possible_num_neurons,
                    possible_dropouts, activations_function, output_activation,
                    possible_attention_heads, attention_dropout, 
                    possible_lrs, loss_function_weights[cancer_type])

      gat_tuner = kt.Hyperband(gat_model_builder,
                     objective=kt.Objective('auc_pr', direction='max'),
                     max_epochs=hp_epochs,
                     directory=dm.base_path,
                     project_name='hp_tuner_gat')

      print("\nTuning GAT model...")
      with suppress_stdout():
        gat_tuner.search(train_gen, epochs=hp_epochs, shuffle=False,
          validation_data=(validation_gen), callbacks=[hp_early_stop])
        
        best_hps = gat_tuner.get_best_hyperparameters(5)
        # Build the model with the best hp.
        gat_model = gat_model_builder(best_hps[0])

      with open(dm.base_path+'gat_tunner_best_results.txt', 'w') as f:
        with redirect_stdout(f):
          gat_tuner.results_summary()

      print("\nTraining GAT model... ")
      gat_history = gat_model.fit(train_gen, epochs=training_epochs, 
          validation_data=(validation_gen), 
          callbacks=[early_stop], verbose=0, 
          shuffle=False,  # This should be False, since shuffling 
                          # data means shuffling the whole graph
      )
      histories.append(gat_history)

      gat_performance = gat_model.evaluate(test_gen)
      dm.write_to_results_csv(models_names[0], gat_performance)
      gat_pred = gat_model.predict(test_gen)[0]
      preds.append(gat_pred)



      # GCN generators
      generator = FullBatchNodeGenerator(G, method="gcn")
      train_gen = generator.flow(X_train_index, y_train)
      validation_gen = generator.flow(X_validation_index, y_validation)
      test_gen = generator.flow(X_test_index, y_test)

      gcn_model_builder = get_gcn_model(generator, y_train.shape[1], 
                    possible_num_layers, possible_gammas, possible_num_neurons,
                    possible_dropouts, activations_function, output_activation,
                    possible_lrs, loss_function_weights[cancer_type])

      gcn_tuner = kt.Hyperband(gcn_model_builder,
                     objective=kt.Objective('auc_pr', direction='max'),
                     max_epochs=hp_epochs,
                     directory=dm.base_path,
                     project_name='hp_tuner_gcn')

      print("\nTuning GCN model...")
      with suppress_stdout():
        gcn_tuner.search(train_gen, epochs=hp_epochs, shuffle=False,
          validation_data=(validation_gen), callbacks=[hp_early_stop])
        
        best_hps = gcn_tuner.get_best_hyperparameters(5)
        # Build the model with the best hp.
        gcn_model = gcn_model_builder(best_hps[0])

      with open(dm.base_path+'gcn_tunner_best_results.txt', 'w') as f:
        with redirect_stdout(f):
          gcn_tuner.results_summary()

      print("\nTraining GCN model... ")
      gcn_history = gcn_model.fit(train_gen, epochs=training_epochs, 
          validation_data=(validation_gen), 
          callbacks=[early_stop], verbose=0, 
          shuffle=False,  # This should be False, since shuffling 
                          # data means shuffling the whole graph
      )
      histories.append(gcn_history)

      gcn_performance = gcn_model.evaluate(test_gen)
      dm.write_to_results_csv(models_names[1], gcn_performance)
      gcn_pred = gcn_model.predict(test_gen)[0]
      preds.append(gcn_pred)


      
      if train_mlp:
        mlp_model_builder = get_mlp_model(y_train.shape[1], 
                              X_train.shape[1], possible_num_layers,
                              possible_gammas, possible_num_neurons, 
                              possible_dropouts, activations_function,
                              output_activation, possible_lrs, 
                              loss_function_weights[cancer_type])

        mlp_tuner = kt.Hyperband(mlp_model_builder,
                     objective=kt.Objective('auc_pr', direction='max'),
                     max_epochs=hp_epochs,
                     directory=dm.base_path,
                     project_name='hp_tuner_mlp')

        print("\nTuning MLP model...")
        with suppress_stdout():
          mlp_tuner.search(X_train, y_train, epochs=hp_epochs, 
            validation_data=(X_validation, y_validation),
            callbacks=[hp_early_stop], batch_size=mlp_batch_size)

          best_hps = mlp_tuner.get_best_hyperparameters(5)
          # Build the model with the best hp.
          mlp_model = mlp_model_builder(best_hps[0])

        with open(dm.base_path+'mlp_tunner_best_results.txt', 'w') as f:
          with redirect_stdout(f):
            mlp_tuner.results_summary()

        print("\nTraining MLP model...")
        mlp_history = mlp_model.fit(X_train, y_train, epochs=training_epochs, 
                validation_data=(X_validation, y_validation), 
                callbacks=[early_stop], batch_size=mlp_batch_size, verbose=0)
        histories.append(mlp_history)

        mlp_performance = mlp_model.evaluate(X_test, y_test)
        dm.write_to_results_csv(models_names[2], mlp_performance)
        mlp_pred = mlp_model.predict(X_test)
        preds.append(mlp_pred)



      print("\nPlotting models' history (loss, acc, auc_roc, pr_auc) ...")
      dm.save_plots_history(histories, i)
      
      print("\nSaving confusion matrices...\n")
      dm.save_conf_matrices(preds, y_test, i)
