import numpy as np
import tensorflow as tf
import random as rn
import keras
import os,sys
import pickle as pkl
os.environ['KERAS_BACKEND'] = 'tensorflow'
base_fp = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.insert(0, base_fp) #allows word_vec_utils to be imported
import argparse
from models.prototypes.baseline import get_model
from keras.callbacks import TensorBoard
from utils.data_utils import CocoCaptions
from utils.data_utils import get_data
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

    Captions = CocoCaptions(args.data,args.max_samples)
    WV = FilteredGloveVectors()
    Captions.initialize_WV(WV)

    Captions,ValCaptions = Captions.split_train_val(args.train_val_split)
    print("Train Len",len(Captions))
    print("Val Len",len(ValCaptions))

    embedding_matrix = WV.get_embedding_matrix()
    metrics = Metrics()

    hp_searcher = HPSearcher([0.0001], 
                            embedding_matrix, 
                            args.model, 
                            Captions, 
                            custom_metrics = [metrics], 
                            max_samples=args.max_samples, 
                            path_load_model=args.load, 
                            val_helper=ValCaptions,
                            epochs=args.epochs,
                            gen = args.gen,
                            gpu = args.gpu,
                            batch_size=args.batch_size)
    results = hp_searcher.run()

def encode(args):

    Captions = CocoCaptions(args.data,args.max_samples)
    WV = FilteredGloveVectors()
    Captions.initialize_WV(WV)
    embedding_matrix = WV.get_embedding_matrix()

    if args.model[:4] == "cap2" or args.model[:4] == "vae2" :
            inputs, outputs = None, None
            datagen, valgen = None, None
            cap2 = None

            hparams = HParams(learning_rate=args.learning_rate, hidden_dim=1024,
                        optimizer='adam', dropout= 0.5, 
                        max_seq_length=Captions.max_caption_len,
                        embed_dim=embedding_matrix.shape[-1],
                        num_embeddings=embedding_matrix.shape[0],
                        activation='relu',
                        latent_dim=1000)

            if args.gen == 'train' or args.gen == 'all':
                data = get_data(args.model, Captions, gen=True)
            else:
                data = get_data(args.model, Captions)

            ModelClass = get_model(args.model)
            model = ModelClass(hparams, embeddings=embedding_matrix)
            
            if args.load is not None:
                print("Loading model "+args.load+" ...")
                model.load_model(args.load)
            
            model.compile()

            encoder = model.get_encoder()

            
            if isinstance(data, keras.utils.Sequence):
                preds = encoder.predict_generator(data,verbose=1)
            
            elif isinstance(data, tuple):
                preds = encoder.predict(x=data[0],verbose=1)
            
            
            X = Captions.ordered_IDs
            print("ordered_X1",len(X)," ")
            
            new_X = []
            for image_id in X:
                captions = Captions.get_captions(image_id)
                X_group, Y_group = Captions.get_caption_convolutions(captions,False)
                for c,_ in zip(X_group,Y_group):
                    new_X.append((c,image_id))

            print("ordered_X2",len(new_X)," ")
            X = new_X[len(preds):]
            print("Predicted ",len(preds)," preds")
            print("ordered_X2",len(new_X)," ")
            
            
            output = []

            for (c,image_id),y in zip(X,preds):
                sentence = Captions.WV.indices_to_words(c)
                sentence = " ".join(sentence[1:-1])
                resnet = Captions.get_resnet_output(image_id)
                output.append((sentence,resnet,y))

            print("Predicted ",len(output)," outputs")

                

            save_loc = base_fp+"/skip-thoughts/our_model_encodings.pkl"
            pkl.dump(output,open(save_loc,"wb+"),2)
            print("Output saved")





def train(args):
    Captions = CocoCaptions(args.data,args.max_samples)
    WV = FilteredGloveVectors()
    Captions.initialize_WV(WV)

    Captions,ValCaptions = Captions.split_train_val()

    embedding_matrix = WV.get_embedding_matrix()
    metrics = Metrics()

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(args.learning_rate))
    print()
    
    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(args.learning_rate, args.model)
    
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

    if args.model == 'toy':

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
                        callbacks=[callback_log]+[metrics])
    else:
        if args.model[:4] == "cap2" or args.model[:4] == "vae2" :
            inputs, ordered_outputs = None, None
            datagen, valgen = None, None
            cap2 = None
            callbacks = [callback_log]

            hparams = HParams(learning_rate=args.learning_rate, hidden_dim=1024,
                        optimizer='adam', dropout= 0.5, 
                        max_seq_length=Captions.max_caption_len,
                        embed_dim=embedding_matrix.shape[-1],
                        num_embeddings=embedding_matrix.shape[0],
                        activation='relu',
                        latent_dim=1000)

            if args.gen == 'train' or args.gen == 'all':
                data = get_data(args.model, Captions, gen=True)
                if args.gen == 'all':
                    val_data = get_data(args.model, ValCaptions, gen=True)
                else:
                    val_data = get_data(args.model, ValCaptions)
            else:
                data = get_data(args.model, Captions)
                val_data = get_data(args.model, ValCaptions)

            if args.model is not 'cap2img':
                metrics.validation_data = val_data
                callbacks += [metrics]

            ModelClass = get_model(args.model)
            model = ModelClass(hparams, embeddings=embedding_matrix)
             
            
            if args.load is not None:
                print("Loading model "+args.load+" ...")
                model.load_model(args.load)
            
            model.compile()
            
            if isinstance(data, keras.utils.Sequence):
                history = model.model.fit_generator(data,
                            epochs=args.epochs,
                            validation_data=val_data,
                            callbacks=callbacks,
                            )
            elif isinstance(data, tuple):
                history = model.model.fit(x=data[0],
                                y=data[1],
                                epochs=args.epochs,
                                validation_data=val_data,
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


def main(args):
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
    config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically

    from keras import backend as K
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    if args.encode():
        encode(args)
    else:
        train(args)

    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='test', action='store_true')
    parser.add_argument('--model', help='model name', default='cap2all')
    parser.add_argument('--data', help='data', type=int, default=3)
    parser.add_argument('-hp', help='perform hyper parameter search', action='store_true')
    parser.add_argument('--max_samples', help='maximum number of data samples to train on', type=int, default=None)
    parser.add_argument('--load', help='load saved model')
    parser.add_argument('--path', help='save path', default='')
    parser.add_argument('--epochs', help='number of epochs', type=int, default=1)
    parser.add_argument('-g','--gen',help='whether to use generator for train or both train and val')
    parser.add_argument('-lr','--learning_rate', help='learning_rate', type=int, default=0.00001)
    parser.add_argument('-e', '--encode',help='whether to simply encode captions', action='store_true')
    parser.add_argument('--gpu', help='whether to use gpu or cpu', type=int, default=0)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=32)
    parser.add_argument('--train_val_split',help='determine percent train [0,1]',type=float,default=0.7)


    args = parser.parse_args()

    if args.t is False:
        if args.hp:
            hp_search(args)
        elif args.encode:
            encode(args)
        else:
            main(args)

    else:    
        ## for running any tests
        hparams = HParams()
        print(type(hparams) is HParams)
