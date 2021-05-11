from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

from tensorflow.keras import layers, optimizers, losses, metrics, Model

from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT

def get_gat_model(generator, output_dimention,
                  first_layer_size=64, second_layer_size=64, 
                  first_activation='elu', second_activation='elu', output_activation='softmax',
                  attention_heads=8, in_dropout=0.15, attention_dropout=0.15, 
                  learning_rate=0.0001, loss_function='categorical_crossentropy'):

    #generator = FullBatchNodeGenerator(G, method="gat")
    #train_gen = generator.flow(train_subjects.index, train_targets)

    gat = GAT(
        layer_sizes=[first_layer_size, second_layer_size, output_dimention],
        activations=[first_activation, second_activation, output_activation],
        attn_heads=attention_heads,
        generator=generator,
        in_dropout=in_dropout,
        attn_dropout=attention_dropout,
        normalize=None,
    )

    x_inp, predictions = gat.in_out_tensors()
    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=loss_function,
            metrics=["acc", metrics.AUC(curve="ROC", name="auc_roc"), metrics.AUC(curve="PR", name="auc_pr")])

    return model


def get_mlp_model(number_of_classes, input_dimension, 
                  num_neurons_1st_layer=64, num_neurons_2nd_layer=64,
                  dropout=0.15,
                  first_activation='elu', second_activation='elu', output_activation='softmax',
                  learning_rate=0.0001, loss_function='categorical_crossentropy'):

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

    model.compile(loss=loss_function, optimizer=optimizers.Adam(lr=learning_rate), 
              metrics=["acc", metrics.AUC(curve="ROC", name="auc_roc"), metrics.AUC(curve="PR", name="auc_pr")])
    
    return model
