import numpy as np
import keras
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support

class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = None
        if isinstance(self.validation_data, keras.utils.Sequence): 
            preds = self.model.predict_generator(self.validation_data)
        else:
            if isinstance(self.validation_data, list):
                preds = preds = self.model.predict([self.validation_data[0], self.validation_data[1]])
        for item in preds:
            if len(item.shape) == 3:
                val_predict = (np.asarray(item))
                break
        val_predict = np.argmax(val_predict, axis=2)
        val_targ=None
        for item in self.validation_data:
            if len(item.shape)==3:
                val_targ = item[:,:,0]
                break
        val_predict_flat = val_predict.flatten()
        val_targ_flat = val_targ.flatten()
        val_mask = val_targ_flat != 0
        val_predict_flat = val_predict_flat[val_mask]
        val_targ_flat = val_targ_flat[val_mask]
        _val_precision, _val_recall, _val_f1, _ = precision_recall_fscore_support(val_targ_flat, val_predict_flat, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: {0} — val_precision: {1} — val_recall {2}".format(_val_f1, _val_precision, _val_recall))
        return
 
if __name__ == '__main__':
    metrics = Metrics()