from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT, GCN
from losses import categorical_focal_loss

def get_gat_model(generator, output_dimention,
                  possible_num_layers=[1,2,3],
                  possible_gammas=[0,1,2],
                  possible_num_neurons=[32,64,128],
                  possible_in_dropouts=[0.0, 0.1, 0.2, 0.3],
                  activations_function='elu',  output_activation='softmax',
                  possible_attention_heads=[2,4,8], 
                  attention_dropout=0,
                  possible_lrs=[0.0001, 0.0005, 0.001, 0.005],
                  loss_function_weights = None):

    # hp: hyperparameter (tunner)
    def model_builder(hp):

        loss_function = categorical_focal_loss(alpha=loss_function_weights, 
            gamma=hp.Choice("gamma", values=possible_gammas))

        num_neurons_per_layer = []
        num_layers = hp.Choice("num_layers", values=possible_num_layers)
        for layer in range(num_layers):
            num_neurons_per_layer.append(hp.Choice(f"layer_{layer}_units", 
                values=possible_num_neurons))

        in_dropout = hp.Choice("dropout", values=possible_in_dropouts)
        learning_rate = hp.Choice("learning_rate", values=possible_lrs)
        attention_heads = hp.Choice("attention_heads", 
                            values=possible_attention_heads)

        activations = [activations_function]*num_layers
        activations += [output_activation]

        gat = GAT(
            layer_sizes=num_neurons_per_layer+[output_dimention],
            activations=activations,
            attn_heads=attention_heads,
            generator=generator,
            in_dropout=in_dropout,
            attn_dropout=attention_dropout,
            normalize=None
        )

        x_inp, predictions = gat.in_out_tensors()
        model = Model(inputs=x_inp, outputs=predictions)

        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), 
                loss=loss_function,
                metrics=["acc", metrics.AUC(curve="ROC", name="auc_roc"), 
                        metrics.AUC(curve="PR", name="auc_pr"), 
                        metrics.Precision(name="precision"), 
                        metrics.Recall(name="recall")])
        return model

    return model_builder



def get_gcn_model(generator, output_dimention,
                  possible_num_layers=[1,2,3],
                  possible_gammas=[0,1,2],
                  possible_num_neurons=[32,64,128],
                  possible_dropouts=[0.0, 0.1, 0.2, 0.3],
                  activations_function='elu',  output_activation='softmax',
                  possible_lrs=[0.0001, 0.0005, 0.001, 0.005],
                  loss_function_weights = None):

    # hp: hyperparameter (tunner)
    def model_builder(hp):

        loss_function = categorical_focal_loss(alpha=loss_function_weights, 
            gamma=hp.Choice("gamma", values=possible_gammas))

        num_neurons_per_layer = []
        num_layers = hp.Choice("num_layers", values=possible_num_layers)
        for layer in range(num_layers):
            num_neurons_per_layer.append(hp.Choice(f"layer_{layer}_units", 
                values=possible_num_neurons))

        dropout = hp.Choice("dropout", values=possible_dropouts)
        learning_rate = hp.Choice("learning_rate", values=possible_lrs)

        activations = [activations_function]*num_layers
        activations += [output_activation]

        gcn = GCN(
            layer_sizes=num_neurons_per_layer+[output_dimention],
            activations=activations,
            generator=generator,
            dropout=dropout
        )

        x_inp, predictions = gcn.in_out_tensors()
        model = Model(inputs=x_inp, outputs=predictions)

        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), 
                loss=loss_function,
                metrics=["acc", metrics.AUC(curve="ROC", name="auc_roc"), 
                        metrics.AUC(curve="PR", name="auc_pr"), 
                        metrics.Precision(name="precision"), 
                        metrics.Recall(name="recall")])
        return model

    return model_builder



def get_mlp_model(output_dimention, input_dimension, 
                  possible_num_layers=[1,2,3],
                  possible_gammas=[0,1,2],
                  possible_num_neurons=[32,64,128],
                  possible_dropouts=[0.0, 0.1, 0.2, 0.3],
                  activations_function='elu', 
                  output_activation='softmax',
                  possible_lrs=[0.0001, 0.0005, 0.001, 0.005],
                  loss_function_weights = None):

    # hp: hyperparameter (tunner)
    def model_builder(hp):

        input_dim = input_dimension

        loss_function = categorical_focal_loss(alpha=loss_function_weights, 
            gamma=hp.Choice("gamma", values=possible_gammas))

        num_neurons_per_layer = []
        num_layers = hp.Choice("num_layers", values=possible_num_layers)
        for layer in range(num_layers):
            num_neurons_per_layer.append(hp.Choice(f"layer_{layer}_units", 
                values=possible_num_neurons))

        dropout = hp.Choice("dropout", values=possible_dropouts)
        learning_rate = hp.Choice("learning_rate", possible_lrs)

        model = Sequential()
        model.add(Dense(num_neurons_per_layer[0], input_dim=input_dim))
        model.add(Activation(activations_function))
        model.add(Dropout(dropout))

        if len(num_neurons_per_layer) > 1:
            for i in range(1, len(num_neurons_per_layer)):
                model.add(Dense(num_neurons_per_layer[i]))
                model.add(Activation(activations_function))
                model.add(Dropout(dropout))


        model.add(Dense(output_dimention))
        model.add(Activation(output_activation))

        model.compile(loss=loss_function, 
            optimizer=optimizers.Adam(learning_rate=learning_rate), 
            metrics=["acc", metrics.AUC(curve="ROC", name="auc_roc"), 
                        metrics.AUC(curve="PR", name="auc_pr"), 
                        metrics.Precision(name="precision"), 
                        metrics.Recall(name="recall")])
        return model
    
    return model_builder
