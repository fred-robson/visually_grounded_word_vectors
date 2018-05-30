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
from keras.models import Model 
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Embedding

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from models.metrics import Metrics
from models.prototypes.baseline import Cap2
from tensorflow.contrib.training import HParams
from skopt.utils import use_named_args
from datetime import datetime


def _log_dir_name(learning_rate, model):

    date = str(datetime.now()).split(" ")[0]
    # The dir-name for the TensorBoard log-dir.
    s = "./logs/lr_{0:.0e}_model_{1}_on_{2}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       model, date)
    return log_dir

class HPSearcher(object):

    def __init__(self, default_parameters, embedding_matrix, model, data_helper, path_best_model=None):
        self.default_parameters = default_parameters
        self.embedding_matrix = embedding_matrix
        self.model = model
        self.data_helper = data_helper
        self.best_f1 = 0.0
        if path_best_model is None:
            path_best_model = './models/saved/'+model+'_best_model.keras'
        
        self.path_best_model = path_best_model


    def _search_space(self):
        space_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                 name='learning_rate')
        #space_hidden_dim = Integer(low=5, high=512, name='hidden_dim')
        #space_optimizers = Categorical(categories=['adam','sgd'],name='optimizer')
        return [space_learning_rate]

    def run(self):

        @use_named_args(dimensions=self._search_space())
        def _fitness(learning_rate):
            """
            Hyper-parameters:
            learning_rate:     Learning-rate for the optimizer.
            hidden_dim:  Size of Hidden Dimension
            """

            # Print the hyper-parameters.
            print('learning rate: {0:.1e}'.format(learning_rate))
            print()
            
            # Dir-name for the TensorBoard log-files.
            log_dir = _log_dir_name(learning_rate, self.model)
            
            # Create a callback-function for Keras which will be
            # run after each epoch has ended during training.
            # This saves the log-files for TensorBoard.
            # Note that there are complications when histogram_freq=1.
            # It might give strange errors and it also does not properly
            # support Keras data-generators for the validation-set.
            callback_log = TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
                batch_size=32,
                write_graph=True,
                write_grads=False,
                write_images=False)

            model = None
            history = None
            validation_data=None
            # Create the neural network with these hyper-parameters.

            if self.model == 'toy':

                X = np.random.randint(0, 6, size=(3000,50))
                Y = np.random.randint(0, 6, size=(3000,50,1))

                model = Sequential()
                model.add(Embedding(6, 50, input_length=50))
                model.add(Dense(300, activation='relu'))
                model.add(Dense(6, activation='softmax'))
                model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(X,
                                Y,
                                epochs=3,
                                batch_size=128,
                                validation_split=0.2,
                                validation_data=validation_data,
                                callbacks=[callback_log])
            else:
                if self.model[:4] == "cap2":
                    inputs, outputs = None, None

                    if self.model == 'cap2cap':
                        X, Y1, Y2 = self.data_helper.cap2cap()
                        Y2 = np.expand_dims(Y2, axis=2)
                        validation_data=None
                        inputs = {'encoder_input': X, 'decoder_input': Y1}
                        outputs = {'decoder_output': Y2}

                    if self.model == 'cap2img':
                        X, Y = self.data_helper.cap2img()
                        inputs = {'encoder_input': X}
                        outputs = {'resnet_output': Y}

                    if self.model == 'cap2all':
                        X, Y1, Y2, Y3 = self.data_helper.cap2all()
                        inputs = {'encoder_input': X, 'decoder_input': Y1}
                        outputs = {'resnet_output': Y3, 'decoder_output': Y2}

                    hparams = HParams(learning_rate=learning_rate, hidden_dim=1024,
                                optimizer='adam', dropout= 0.5, 
                                max_seq_length=inputs['encoder_input'].shape[1],
                                embed_dim=self.embedding_matrix.shape[-1],
                                num_embeddings=self.embedding_matrix.shape[0])
                    cap2 = Cap2(hparams, model_type=self.model, embeddings=self.embedding_matrix)

                    model = cap2.model
                    history = model.fit(inputs,
                                    outputs,
                                    epochs=3,
                                    batch_size=128,
                                    validation_split=0.2,
                                    validation_data=validation_data,
                                    callbacks=[callback_log])


            # Get the classification accuracy on the validation-set
            # after the last training-epoch.
            
            #f1 = mectrics.val_f1s[-1]
            f1 = 1

            # Print the classification accuracy.
            print()
            print("F1: {0:.2%}".format(f1))
            print()

            # Save the model if it improves on the best-found performance.
            # We use the global keyword so we update the variable outside
            # of this function.
            # If the classification accuracy of the saved model is improved ...
            
            if f1 > self.best_f1:
                # Save the new model to harddisk.
                model.save(self.path_best_model)
                # Update the classification accuracy.
                self.best_f1 = f1

            # Delete the Keras model with these hyper-parameters from memory.
            del model
            
            # Clear the Keras session, otherwise it will keep adding new
            # models to the same TensorFlow graph each time we create
            # a model with a different set of hyper-parameters.
            K.clear_session()
            
            # NOTE: Scikit-optimize does minimization so it tries to
            # find a set of hyper-parameters with the LOWEST fitness-value.
            # Because we are interested in the HIGHEST classification
            # accuracy, we need to negate this number so it can be minimized.
            return -f1

        search_result = gp_minimize(func=_fitness,
                    dimensions=self._search_space(),
                    acq_func='EI', # Expected Improvement.
                    n_calls=40,
                    x0=self.default_parameters)
        return search_result

