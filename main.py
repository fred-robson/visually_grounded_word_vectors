import numpy as np
import tensorflow as tf
import random as rn
import keras 
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import argparse
from keras.callbacks import TensorBoard
from utils.data_utils import CocoCaptions
from utils.word_vec_utils import GloVeVectors
from utils.word_vec_utils import FilteredGloveVectors
from models.metrics import Metrics
from models.prototypes.baseline import Cap2
from tensorflow.contrib.training import HParams
from skopt.utils import use_named_args
from keras.models import Model 
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Embedding

def log_dir_name(learning_rate, model, run_type, dataset):

    date = str(datetime.now()).split(" ")[0]
    # The dir-name for the TensorBoard log-dir.
    s = "./logs/{3}_{4}_lr_{0:.0e}_model_{1}_on_{2}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       model, date, run_type, dataset)
    return log_dir

def hp_search(args):

    from hyperband_skopt import HPSearcher

    Captions = CocoCaptions(args.data)
    WV = FilteredGloveVectors()
    Captions.initialize_WV(WV)

    embedding_matrix = WV.get_embedding_matrix()
    metrics = Metrics()

    hp_searcher = HPSearcher([0.0001], embedding_matrix, args.model, Captions, custom_metrics = [metrics], max_samples=args.max_samples, path_load_model=args.load)
    results = hp_searcher.run()


def main(args):
    print(args)
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(42)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    from keras import backend as K
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    Captions = CocoCaptions(args.data)
    WV = FilteredGloveVectors()
    Captions.initialize_WV(WV)

    embedding_matrix = WV.get_embedding_matrix()
    metrics = Metrics()

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print()
    
    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, args.model)
    
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
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X,
                        Y,
                        epochs=1,
                        batch_size=128,
                        validation_split=0.2,
                        validation_data=validation_data,
                        callbacks=[callback_log]+self.custom_metrics)
    else:
        if self.model[:4] == "cap2":
            inputs, outputs = None, None
            cap2 = None
            callbacks = [callback_log]

            if self.model == 'cap2cap':
                _, X, Y1, Y2 = self.data_helper.cap2cap()
                if self.max_samples is not None:
                    X, Y1, Y2, = X[:self.max_samples], Y1[:self.max_samples], Y2[:self.max_samples]
                Y2 = np.expand_dims(Y2, axis=2)
                validation_data=None
                inputs = {'encoder_input': X, 'decoder_input': Y1}
                outputs = {'decoder_output': Y2}
                hparams = HParams(learning_rate=learning_rate, hidden_dim=1024,
                        optimizer='adam', dropout= 0.5, 
                        max_seq_length=inputs['encoder_input'].shape[1],
                        embed_dim=self.embedding_matrix.shape[-1],
                        num_embeddings=self.embedding_matrix.shape[0],
                        activation='relu',
                        latent_dim=1000)
                cap2 = Cap2Cap(hparams, embeddings=self.embedding_matrix)
                callbacks += self.custom_metrics

            if self.model == 'cap2img':
                _, X, Y = self.data_helper.cap2resnet()
                Y = Y[:,0,:]
                inputs = {'encoder_input': X}
                outputs = {'projection_output': Y}
                hparams = HParams(learning_rate=learning_rate, hidden_dim=1024,
                        optimizer='adam', dropout= 0.5, 
                        max_seq_length=inputs['encoder_input'].shape[1],
                        embed_dim=self.embedding_matrix.shape[-1],
                        num_embeddings=self.embedding_matrix.shape[0],
                        activation='relu',
                        latent_dim=1000)
                cap2 = Cap2Img(hparams, embeddings=self.embedding_matrix)

            if self.model == 'cap2all':
                _, X, Y1, Y2, Y3 = self.data_helper.cap2all()
                X, Y1, Y2, Y3 = X[:20], Y1[:20], Y2[:20], Y3[:20]
                Y2 = np.expand_dims(Y2, axis=2)
                Y3 = Y3[:,0,:]
                if self.max_samples is not None:
                    X, Y1, Y2, Y3 = X[:self.max_samples], Y1[:self.max_samples], Y2[:self.max_samples], Y3[:self.max_samples]
                inputs = {'encoder_input': X, 'decoder_input': Y1}
                outputs = {'projection_output': Y3, 'decoder_output': Y2}
                hparams = HParams(learning_rate=learning_rate, hidden_dim=1024,
                        optimizer='adam', dropout= 0.5, 
                        max_seq_length=inputs['encoder_input'].shape[1],
                        embed_dim=self.embedding_matrix.shape[-1],
                        num_embeddings=self.embedding_matrix.shape[0],
                        activation='relu',
                        latent_dim=1000)
                cap2 = Cap2All(hparams, embeddings=self.embedding_matrix)
                callbacks += self.custom_metrics
            if self.path_load_model is not None:
                print("Loading model "+self.path_load_model+" ...")
                cap2.load_model(self.path_load_model)
            
            cap2.compile()
            model = cap2.model
            # history = model.fit(inputs,
            #                 outputs,
            #                 epochs=3,
            #                 batch_size=128,
            #                 validation_split=0.2,
            #                 validation_data=validation_data,
            #                 callbacks=callbacks)
            history = fit_generator(datagen,
                                    epochs=10,
                                    validation_data=valgen,
                                    callbacks=callbacks,
                                    )


    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    if args.model != 'cap2img':
        f1 = metrics[0].val_f1s[-1]
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
    print("saving model at {0}".format(args.path))
    # Save the new model to harddisk.
    model.save(args.path)
    # Update the classification accuracy.

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='test', action='store_true')
    parser.add_argument('--model', help='model name', default='cap2all')
    parser.add_argument('--data', help='data', type=int, default=3)
    parser.add_argument('-hp', help='perform hyper parameter search', action='store_true')
    parser.add_argument('--max_samples', help='maximum number of data samples to train on', type=int, default=None)
    parser.add_argument('--load', help='load saved model')
    parser.add_argument('--path', help='save path', default='')

    args = parser.parse_args()

    if args.t is False:
        if args.hp:
            hp_search(args)
        else:
            main(args)
    else:    
        ## for running any tests
        hparams = HParams()
        print(type(hparams) is HParams)
