import pandas as pd
import numpy as np
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT

import networkx as nx

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from keras.models import Sequential
from keras.layers import Dense

from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.model_selection import StratifiedKFold





if __name__ =="__main__":
    print("oi")