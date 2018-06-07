import numpy as np
import tensorflow as tf
import keras
from utils.data_utils import get_data

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
from models.prototypes.baseline import Cap2Cap, Cap2Img, Cap2All, Vae2All, get_model
from tensorflow.contrib.training import HParams
from skopt.utils import use_named_args
from datetime import datetime
from tqdm import tqdm


def _log_dir_name(learning_rate, model):

    date = str(datetime.now()).split(" ")[0]
    # The dir-name for the TensorBoard log-dir.
    s = "./logs/lr_{0:.0e}_model_{1}_on_{2}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       model, date)
    return log_dir

class HPSearcher(object):

    def __init__(self, default_parameters, embedding_matrix, model, data_helper, path_best_model=None, path_load_model=None, custom_metrics=[], max_samples=None, val_helper=None, epochs=1, gen=None):
        self.default_parameters = default_parameters
        self.embedding_matrix = embedding_matrix
        self.model = model
        self.data_helper = data_helper
        self.val_helper = val_helper
        self.best_f1 = 0.0
        if path_best_model is None:
            path_best_model = './models/saved/'+model+'_best_model.keras'
        self.custom_metrics = custom_metrics
        self.path_best_model = path_best_model
        self.path_load_model = path_load_model
        self.max_samples = max_samples
        self.epochs=1
        self.gen = gen


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
            #K.clear_session()
            if self.model == 'toy':

                X = np.random.randint(0, 6, size=(3000,50))
                Y = np.random.randint(0, 6, size=(3000,50,1))

                model = Sequential()
                model.add(Embedding(6, 50, input_length=50))
                model.add(Dense(300, activation='relu'))
                model.add(Dense(6, activation='softmax'))
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(X,
                                Y,
                                epochs=1,
                                batch_size=1024,
                                validation_split=0.2,
                                validation_data=validation_data,
                                callbacks=[callback_log]+self.custom_metrics)
            else:
                if self.model[:4] == "cap2" or self.model[:4] == "vae2" :
                    inputs, outputs = None, None
                    datagen, valgen = None, None
                    cap2 = None
                    callbacks = [callback_log]

                    hparams = HParams(learning_rate=learning_rate, hidden_dim=1024,
                                optimizer='adam', dropout= 0.5, 
                                max_seq_length=self.data_helper.max_caption_len,
                                embed_dim=self.embedding_matrix.shape[-1],
                                num_embeddings=self.embedding_matrix.shape[0],
                                activation='relu',
                                latent_dim=1000)

                    if self.gen == 'train' or self.gen == 'all':
                        data = get_data(self.model, self.data_helper, gen=True)
                        if self.gen == 'all':
                            val_data = get_data(self.model, self.val_helper, gen=True)
                        else:
                            val_data = get_data(self.model, self.val_helper)
                    else:
                        data = get_data(self.model, self.data_helper)
                        val_data = get_data(self.model, self.val_helper)
                        # _, X, Y1, Y2 = self.data_helper.cap2cap()
                        # if self.max_samples is not None:
                        #     X, Y1, Y2, = X[:self.max_samples], Y1[:self.max_samples], Y2[:self.max_samples]
                        # Y2 = np.expand_dims(Y2, axis=2)
                        # validation_data=None
                        # inputs = {'encoder_input': X, 'decoder_input': Y1}
                        # outputs = {'decoder_output': Y2}

                    if self.model is not 'cap2img':
                        self.custom_metrics[0].validation_data = val_data
                        callbacks += self.custom_metrics
                        # _, X, Y = self.data_helper.cap2resnet()
                        # Y = Y[:,0,:]
                        # inputs = {'encoder_input': X}
                        # outputs = {'projection_output': Y}

                        # _, X, Y1, Y2, Y3 = self.data_helper.cap2all()
                        # #X, Y1, Y2, Y3 = X[:20], Y1[:20], Y2[:20], Y3[:20]
                        # Y2 = np.expand_dims(Y2, axis=2)
                        # Y3 = Y3[:,0,:]
                        # if self.max_samples is not None:
                        #     X, Y1, Y2, Y3 = X[:self.max_samples], Y1[:self.max_samples], Y2[:self.max_samples], Y3[:self.max_samples]
                        # inputs = {'encoder_input': X, 'decoder_input': Y1}
                        # outputs = {'projection_output': Y3, 'decoder_output': Y2}

                    ModelClass = get_model(self.model)
                    model = ModelClass(hparams, embeddings=self.embedding_matrix)
                     
                    
                    if self.path_load_model is not None:
                        print("Loading model "+self.path_load_model+" ...")
                        model.load_model(self.path_load_model)
                    
                    model.compile()
                    # history = model.fit(inputs,
                    #                 outputs,
                    #                 epochs=3,
                    #                 batch_size=256,
                    #                 validation_split=0.2,
                    #                 validation_data=validation_data,
                    #                 callbacks=callbacks)
                    
                    if isinstance(data, keras.utils.Sequence):
                        history = model.model.fit_generator(data,
                                    epochs=self.epochs,
                                    validation_data=val_data,
                                    callbacks=callbacks,
                                    )
                    elif isinstance(data, tuple):
                        history = model.model.fit(x=data[0],
                                        y=data[1],
                                        epochs=self.epochs,
                                        validation_data=val_data,
                                        callbacks=callbacks,
                                        )



            # Get the classification accuracy on the validation-set
            # after the last training-epoch.
            if self.model != 'cap2img':
                f1 = self.custom_metrics[0].val_f1s[-1]
                print()
                print("Val F1: {0:.2%}".format(f1))
                print()
            else:
                f1 = history.history['val_acc'][-1]
                print()
                print("Val Acc: {0:.2%}".format(f1))
                print()

            # Print the classification accuracy.
            

            # Save the model if it improves on the best-found performance.
            # We use the global keyword so we update the variable outside
            # of this function.
            # If the classification accuracy of the saved model is improved ...
            print(self.best_f1)
            if f1 > self.best_f1:
                print("saving model at {0}".format(self.path_best_model))
                # Save the new model to harddisk.
                model.model.save(self.path_best_model)
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
                    n_calls=10,
                    n_random_starts=4,
                    x0=None,
                    y0 = None)
        return search_result