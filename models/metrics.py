import numpy as np
import keras
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support
from tqdm import tqdm
import gc

def unpack_generator(generator):
	unpack = []
	output_generator = iter(generator)
	gen = [next(output_generator) for _ in range(len(generator))]
	inputs, outputs = zip(*gen)
	inputs = [tuple(i.values()) for i in inputs]
	for i in zip(*inputs):
		array=[]
		arrays = np.asarray(i)
		for j in range(arrays.shape[0]):
			if j == 0: 
				array = arrays[j]
			else:
				array = np.concatenate((array, arrays[j]), axis=0)
		unpack += [array]
	outputs = [tuple(o.values()) for o in outputs]
	for i in zip(*outputs):
		array=[]
		arrays = np.asarray(i)
		for j in range(arrays.shape[0]):
			if j == 0:
				array = arrays[j]
			else:
				array = np.concatenate((array, arrays[j]), axis=0)
		unpack += [array]
	return unpack

class Metrics(Callback):

	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []

	def on_epoch_end_old(self, epoch, logs={}):
		val_predict = None
		if isinstance(self.validation_data, keras.utils.Sequence): 
			preds = self.model.predict_generator(self.validation_data,steps=len(self.validation_data),verbose=1, workers=4,use_multiprocessing=True)
		else:
			if isinstance(self.validation_data, list):
				preds = self.model.predict([self.validation_data[0], self.validation_data[1]], verbose=1)
		for item in list(preds):
			if len(item.shape) == 3:
				val_predict = (np.asarray(item))
				break
		val_predict = np.argmax(val_predict, axis=2)
		val_targ=None
		if isinstance(self.validation_data, keras.utils.Sequence):
			validation_data = unpack_generator(self.validation_data)
		else:
			validation_data = self.validation_data
		for item in validation_data:
			if len(item.shape)==3:
				val_targ = item[:,:,0]
				break
		print("**********",val_predict.shape)
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

	def on_epoch_end(self, epoch, logs={}):
		
		

		def get_val_predict(batch_preds):
			#Get predicted models from @batch of the dataset
			val_predict_batch = None
			for item in list(batch_preds):
				print(item.shape)
				if len(item.shape) == 3:
					val_predict_batch = (np.asarray(item))
					break
			val_predict_batch = np.argmax(val_predict_batch, axis=2)
			return val_predict_batch

		
		def get_val_predict_gen(generator):
			#generator is some form of data [{x_batch1},{y_batch1},{x_batch2},{y_batch2}]
			val_predict = None
			for i,(x,y) in tqdm(enumerate(generator),desc="Forward propogating on generator",total=len(generator)):
				if i==0: 
					val_predict = get_val_predict(self.model.predict_on_batch(x))
				else: 
					val_predict_batch = get_val_predict(self.model.predict_on_batch(x))
					val_predict = np.concatenate((val_predict,val_predict_batch))
				if i == len(generator)-1: break

			return val_predict

		def get_val_predict_list(x_batched,y_batched):
			val_predict = None
			for i,(x_b,y_b) in tqdm(enumerate(zip(x_batched,y_batched)),desc="Forward propogating on data"):
				if i==0: 
					val_predict = get_val_predict(self.model.predict([x_b,y_b],verbose=0))
				else: 
					val_predict_batch = get_val_predict(self.model.predict([x_b,y_b],verbose=0))
					val_predict = np.concatenate((val_predict,val_predict_batch))
				return val_predict

		def get_val_targ():
			#Get true values from dataset
			val_targ=None
			if isinstance(self.validation_data, keras.utils.Sequence):
				validation_data = unpack_generator(self.validation_data)
			else:
				validation_data = self.validation_data
			
			for item in validation_data:
				if len(item.shape)==3:
					val_targ = item[:,:,0]
					break
			return val_targ



		val_predict = None
		val_targ = None
		if isinstance(self.validation_data, keras.utils.Sequence): 
			val_predict = get_val_predict_gen(self.validation_data)
			
			
		elif isinstance(self.validation_data, list):
			x,y = self.validation_data[0], self.validation_data[1]
			batch_size = 128
			num_batches = x.shape[0] / batch_size #TO FIX: Need to link to actual batch size
			indices = [i*batch_size for i in range(int(num_batches))]
			x_batched = np.array_split(x,indices)
			y_batched = np.array_split(y,indices)
			val_predict = get_val_predict_list(x_batched,y_batched)
			
		val_targ = get_val_targ()

		

		print("TARG:",val_targ.shape)
		print("PREDICT",val_predict.shape)

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