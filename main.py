import numpy as np
import tensorflow as tf
import random as rn
import keras 
import os
os.environ['KERAS_BACKEND'] = 'theano'
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

def hp_search(args):

    from hpsearch.hyperband_skopt import HPSearcher

    Captions = CocoCaptions(args.data)
    WV = FilteredGloveVectors()
    Captions.initialize_WV(WV)

    embedding_matrix = WV.get_embedding_matrix()
    metrics = Metrics()

    hp_searcher = HPSearcher([0.0001], embedding_matrix, args.model, Captions, custom_metrics = [metrics], max_samples=args.max_samples)
    results = hp_searcher.run()


def main():
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='test', action='store_true')
    parser.add_argument('--model', help='model name', default='cap2all')
    parser.add_argument('--data', help='data', type=int, default=3)
    parser.add_argument('-hp', help='perform hyper parameter search', action='store_true')
    parser.add_argument('--max_samples', help='maximum number of data samples to train on', type=int, default=None)
    args = parser.parse_args()

    if args.t is False:
        if args.hp:
            hp_search(args)
        else:
            tf.app.run()
    else:    
        ## for running any tests
        hparams = HParams()
        print(type(hparams) is HParams)
