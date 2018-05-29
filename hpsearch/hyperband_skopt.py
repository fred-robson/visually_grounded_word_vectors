import numpy as np
import tensorflow as tf
import keras

from keras import backend as K
from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
<<<<<<< HEAD
>>>>>>> e4af9544b71e260081ee55e72d983efee44448c8
from skopt.utils import use_named_args


class HPSearcher(object):

    def __init__(self, default_parameters):
        self.default_parameters = default_parameters

<<<<<<< HEAD
    def _search_space(self):
        space_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                 name='learning_rate')
        space_hidden_dim = Integer(low=5, high=512, name='hidden_dim')
        space_optimizers = Categorical(categories=['adam','sgd'],name='optimizer')
        return [space_learning_rate, space_hidden_dim, space_optimizers]

    def run(self, func):
        search_result = gp_minimize(func=func,
                    dimensions=self._search_space(),
                    acq_func='EI', # Expected Improvement.
                    n_calls=40,
                    x0=self.default_parameters)
>>>>>>> e4af9544b71e260081ee55e72d983efee44448c8
        return search_result

