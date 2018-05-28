import numpy as np
import tensorflow as tf
import random as rn
import keras 
import os
import argparse
from keras.callbacks import TensorBoard
from utils.data_utils import CocoCaptions
from utils.data_utils import GloVeVectors

def hp_search(model_type):

    from hpsearch.hyperband_skopt import HPSearcher

    def _log_dir_name(learning_rate, hidden_dim):

        # The dir-name for the TensorBoard log-dir.
        s = "./logs/lr_{0:.0e}_hidden_dim_{1}/"

        # Insert all the hyper-parameters in the dir-name.
        log_dir = s.format(learning_rate,
                           hidden_dim)
        return log_dir

    @use_named_args(dimensions=dimensions)
    def _fitness(learning_rate, hidden_dim, optimizer):
        """
        Hyper-parameters:
        learning_rate:     Learning-rate for the optimizer.
        hidden_dim:  Size of Hidden Dimension
        """

        # Print the hyper-parameters.
        print('learning rate: {0:.1e}'.format(learning_rate))
        print('hidden_dim:', hidden_dim)
        print()
        
        # Create the neural network with these hyper-parameters.
        if model_type[:4] == "cap2":
            from models.prototypes.baseline import Cap2
            params = {'learning_rate':learning_rate, 'hidden_dim':hidden_dim,
                 'optimizer':optimizer, 'dropout': 0.5}
            model = Cap2(params, model_type=model_type)

        # Dir-name for the TensorBoard log-files.
        log_dir = _log_dir_name(learning_rate, hidden_dim)
        
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
       
        # Use Keras to train the model.
        history = model.fit(x=data.train.images,
                            y=data.train.labels,
                            epochs=3,
                            batch_size=128,
                            validation_data=validation_data,
                            callbacks=[callback_log])

        # Get the classification accuracy on the validation-set
        # after the last training-epoch.
        accuracy = history.history['val_acc'][-1]

        # Print the classification accuracy.
        print()
        print("Accuracy: {0:.2%}".format(accuracy))
        print()

        # Save the model if it improves on the best-found performance.
        # We use the global keyword so we update the variable outside
        # of this function.
        global best_accuracy

        # If the classification accuracy of the saved model is improved ...
        if accuracy > best_accuracy:
            # Save the new model to harddisk.
            model.save(path_best_model)
            # Update the classification accuracy.
            best_accuracy = accuracy

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
        return -accuracy

    results = HPSearcher.run(_fitness)


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
    parser.add_argument('-t','--test', help='test')
    parser.add_argument('--model', help='model name', default='cap2all')
    parser.add_argument('--data', help='data', type=int)
    parser.add_argument('--hp', help='perform hyper parameter search', action='store_true')
    args = parser.parse_args()

    if args.test is None:
        if args.hp:
            hp_search(args.model)
        else:
            tf.app.run()
    else:    
        ## for running any tests
        caps = CocoCaptions(WV_type=GloVeVectors)
        vocab = caps.get_vocab()