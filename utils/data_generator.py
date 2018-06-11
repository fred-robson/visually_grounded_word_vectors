import keras
import numpy as np

#See here for more information 
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_ids, __data_generation,batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.image_ids = image_ids
        self.shuffle = shuffle
        self.__data_generation = __data_generation
        self.on_epoch_end()
        keras.utils.Sequence.__init__(self)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.image_ids[k] for k in indexes]

        # Generate data
        ret = self.__data_generation(list_IDs_temp)

        return ret

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.Captions.ordered_X = []
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


if __name__ == "__main__":
    pass
