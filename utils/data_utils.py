#Classes and functions for accessing stuff from data files
import json, os, sys

base_fp = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.insert(0, base_fp) #allows word_vec_utils to be imported

from collections import defaultdict
from nltk.tokenize import word_tokenize as tokenize
import scipy.misc
from tqdm import tqdm
import pickle as pkl
import numpy as np
from word_vec_utils import GloVeVectors,CaptionGloveVectors,pad,unk
from keras.preprocessing.sequence import pad_sequences






class CocoCaptions():

	def __init__(self,data=3):
		'''
		Data lets you know where to pull the captions from.
			0 = train
			1 = val
			2 = train and val
			3 = tiny (Note that tiny is a subset of train)

		WV is the kind of word vectors you want to use. Is only necessary for Loren's stuff
		for now 
		'''
		val_file   = base_fp+"../data/Coco/Annotations/captions_val2014.json"
		train_file = base_fp+"../data/Coco/Annotations/captions_train2014.json" 
		tiny_file = base_fp+"../data/Coco/Annotations/captions_tiny2014.json" 

		self.data_source = {train_file:"train",val_file:"val",tiny_file:"tiny"}

		data_options = {0:[train_file],1:[val_file],2:[train_file,val_file],3:[tiny_file]}

		self.save_loc = base_fp+"saved_items/CocoCaptions_saved_%i.pkl"%data
		
		self.data = self.create_data(data_options[data]) #{(image_id,source):[caption1,caption2,....]}
		self.max_caption_len = self.get_longest_caption()+2 #Plus two for start and end tokens

		self.WordVectors = None

		self.image_locations = {"train":base_fp+"../data/Coco/Train2014/COCO_train2014_",
							    "val":base_fp+"../data/Coco/Val2014/COCO_val2014_",
							    "tiny":base_fp+"../data/Coco/Tiny2014/COCO_train2014_",
							   }
		self.resnet_locations ={"train":base_fp+"../data/Coco_ResNet/Train2014/COCO_train2014_",
							    "val":base_fp+"../data/Coco_ResNet/Val2014/COCO_val2014_",
							    "tiny":base_fp+"../data/Coco_ResNet/Tiny2014/COCO_train2014_",
							   }
		#Will not initialize WV unless necessary 
		self.WV = None

	##########
	# Set Up #
	##########

	def create_data(self,file_names):
		'''
		Loads data in the format: {(image_id,source):[caption1,caption2,....]}
		'''
		if os.path.isfile(self.save_loc):
			return pkl.load(open(self.save_loc,'rb')) 

		data = defaultdict(lambda:[])
		loaded_jsons = []

		for f in file_names: 
			loaded_jsons=self.load_json(f)

		
			print("Running one-time tokenization of captions")
			for v in tqdm(loaded_jsons):
				data_source = self.data_source[f]
				image_id = int(v['image_id']),data_source
				data[image_id].append(tokenize(v['caption'].lower()))

		data = dict(data) #remove default dict

		pkl.dump(data,open(self.save_loc,"wb"))

		return data

	def load_json(self,file_name):
		with open(file_name) as f:
			data = json.load(f)
		return data['annotations']

	#################
	# Caption Stuff #
	#################

	def get_vocab(self):
		vocab = set()
		for v in self.data:
			vocab.add(v)
		return vocab

	def get_captions(self,image_id):
		#Image id is of the form (id,source) eg (1423,"val")
		return self.data[image_id]

	def get_all_captions(self):
		keys = list(self.data.keys())
		i = 0
		len_keys = len(keys)
		while i < len_keys:
			k = keys[i]
			yield self.get_captions(k),k
			i+=1

	def build_corpus(self):
		'''
		Builds corpus for GloVe to run on
		'''
		outfile = "GloVe/corpus"
		with open(outfile,"w+") as out: 
			for captions in self.data.values():
				for caption in captions:
					for c in caption:
						out.write(c) 
						out.write(" ")
					out.write("\n")

	def get_longest_caption(self):
		lengths = []
		for captions,_ in self.get_all_captions():
			lengths += [len(c) for c in captions]
		return max(lengths)

	###############
	# Image Stuff #
	###############

	def get_image_file_address(self,image_id):
		#Image id is of the form (id,source) eg (1423,"val")
		image_num = image_id[0]
		data_source = image_id[1]
		file_path = self.image_locations[data_source]
		elongated_id = (12-len(str(image_num)))*"0"+str(image_num) #Length of digits is 12 in folder
		file_path+=elongated_id+".jpg"
		return file_path

	def get_image(self,image_id):
		'''
		Image id is of the form (id,source) eg (1423,"val")
		Returns a matrix representing the RGB vals of the image
		'''
		if not image_id in self.data:
			raise Exception("Image not in dataset")
		address = self.get_image_file_address(image_id)
		return scipy.misc.imread(address, flatten=False, mode='RGB')

	def get_all_images(self):
		'''
		Loads all the images using a generator
		'''
		keys = list(self.data.keys())
		i = 0
		len_keys = len(keys)
		while i < len_keys:
			k = keys[i]
			yield self.get_image(k),k
			i+=1

	################
	# Resnet stuff #
	################

	def get_image_resnet_address(self,image_id):
		'''
		Get the path to the resnet output
		'''
		image_num = image_id[0]
		data_source = image_id[1]
		file_path = self.resnet_locations[data_source]
		file_path+=str(image_id[0])+".npy"
		return file_path

	def get_resnet_output(self,image_id):
		#Returns the resnet output in the form of a numpy array 
		file_path = self.get_image_resnet_address(image_id)
		return np.load(file_path)

	def get_all_resnet_output(self,image_id):
		keys = list(self.data.keys())
		i = 0
		len_keys = len(keys)
		while i < len_keys:
			k = keys[i]
			yield self.get_resnet_output(k),k
			i+=1

	########
	# Misc #
	########

	def num_images(self):
		return len(self.data)

	#####################
	# Loren Model stuff #
	#####################
	def initialize_WV(self,WV):
		self.WV = WV

	def pad_sequences(self,word_indices):
		ret = pad_sequences([word_indices],
							maxlen=self.max_caption_len,
							padding='post', 
							truncating='post',
							value=self.WV.w2i[pad])
		return ret.flatten()

	def get_caption_convolutions(self,captions):
		'''
		Given a set of n captions returns a pair of lists where for each 
		caption c1, there will be 4 pairs (c1,c2) where c2 is in captions but
		not c1
		'''
		X,Y = [],[]
		for c in captions:
				for o in captions:
					if o!=c: 
						x = self.WV.words_to_indices(c)
						y = self.WV.words_to_indices(o)
						x = self.pad_sequences(x)
						y = self.pad_sequences(y)
						X.append(x)
						Y.append(y)
		return X,Y

	def cap2cap(self):
		'''
		Returns X,Y where each x_i is a list of indices of a caption
		and each y_i is a list of indices of a different caption 
		corresponding to the same image
		'''
		if self.WV is None: raise "Call initialize_WV() first"

		X,Y1,Y2 = [],[]

		for captions,image_id in self.get_all_captions():
			X_batch, Y_batch = self.get_caption_convolutions(captions)
			for x,y in zip(X_batch,Y_batch):
				X.append(x)

			
		return np.array(X),np.array(Y1),np.array(Y2)

	def cap2resnet(self):
		if self.WV is None: raise "Call initialize_WV() first" 

		X,Y = [],[]

		for captions,image_id in self.get_all_captions():
			for c in captions:
				x = self.WV.words_to_indices(c)
				x = self.pad_sequences(x)
				X.append(self.WV.words_to_indices(c))
				Y.append(self.get_resnet_output(image_id))

		return np.array(X),np.array(Y)

	def cap2all(self):

		if self.WV is None: raise "Call initialize_WV() first"

		X,Y = [],[]

		for captions,image_id in self.get_all_captions():
			X_batch, Y_batch = self.get_caption_convolutions(captions)
			resnet = self.get_resnet_output(image_id)
			
			for x,y in zip(X_batch,Y_batch):
				X.append(x)
				Y.append((resnet,y))

		return np.array(X),np.array(Y)



if __name__ == "__main__":
	
	'''
	CG = CaptionGloveVectors()
	G = GloVeVectors()
	'''
	Captions = CocoCaptions(3)
	WV = CaptionGloveVectors()
	Captions.initialize_WV(WV)
	X,Y = Captions.cap2all()
	for y in Y: print(Captions.WV.indices_to_words(y[1],remove_padding=True))

