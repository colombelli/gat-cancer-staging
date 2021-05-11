from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

def get_gat_model(df_patients, df_features):
    return

def get_mlp_model(number_of_classes, input_dimension, 
                  num_neurons_1st_layer=64, num_neurons_2nd_layer=64,
                  dropout=0.15,
                  first_activation='elu', second_activation='elu', output_activation='softmax'):
    nb_classes = number_of_classes
    input_dim = input_dimension

    model = Sequential()
    model.add(Dense(num_neurons_1st_layer, input_dim=input_dim))
    model.add(Activation(first_activation))
    model.add(Dropout(dropout))
    model.add(Dense(num_neurons_2nd_layer))
    model.add(Activation(second_activation))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes))
    model.add(Activation(output_activation))

    