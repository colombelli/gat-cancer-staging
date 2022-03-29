from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT

def get_gat_model(generator, output_dimention,
                  num_neurons_per_layer=[64,64],
                  activations=['elu', 'elu'],  output_activation='softmax',
                  attention_heads=8, in_dropout=0.15, attention_dropout=0.15, 
                  learning_rate=0.0001, loss_function='categorical_crossentropy'):

    gat = GAT(
        layer_sizes=num_neurons_per_layer+[output_dimention],
        activations=activations+[output_activation],
        attn_heads=attention_heads,
        generator=generator,
        in_dropout=in_dropout,
        attn_dropout=attention_dropout,
        normalize=None,
    )

    x_inp, predictions = gat.in_out_tensors()
    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=loss_function,
            metrics=["acc", metrics.AUC(curve="ROC", name="auc_roc"), 
                    metrics.AUC(curve="PR", name="auc_pr"), 
                    metrics.Precision(name="precision"), 
                    metrics.Recall(name="recall")])

    return model


def get_mlp_model(output_dimention, input_dimension, 
                  num_neurons_per_layer=[64,64],
                  dropout=0.15, activations=['elu', 'elu'], 
                  output_activation='softmax',
                  learning_rate=0.0001, 
                  loss_function='categorical_crossentropy'):

    input_dim = input_dimension

    model = Sequential()
    model.add(Dense(num_neurons_per_layer[0], input_dim=input_dim))
    model.add(Activation(activations[0]))
    model.add(Dropout(dropout))

    if len(num_neurons_per_layer) > 1:
        for i in range(1, len(num_neurons_per_layer)):
            model.add(Dense(num_neurons_per_layer[i]))
            model.add(Activation(activations[i]))
            model.add(Dropout(dropout))


    model.add(Dense(output_dimention))
    model.add(Activation(output_activation))

    model.compile(loss=loss_function, optimizer=optimizers.Adam(learning_rate=learning_rate), 
          metrics=["acc", metrics.AUC(curve="ROC", name="auc_roc"), 
                    metrics.AUC(curve="PR", name="auc_pr"), 
                    metrics.Precision(name="precision"), 
                    metrics.Recall(name="recall")])
    
    return model
