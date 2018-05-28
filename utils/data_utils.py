#Classes and functions for accessing stuff from data files
import json, os
from collections import defaultdict
from nltk.tokenize import word_tokenize as tokenize
import scipy.misc
from tqdm import tqdm
import pickle as pkl
import numpy as np

base_fp = os.path.dirname(os.path.abspath(__file__))+"/"


class CocoCaptions():

	def __init__(self,data=3):
		'''
		Data lets you know where to pull the captions from.
			0 = train
			1 = val
			2 = train and val
			3 = tiny (Note that tiny is a subset of train)
		'''
		val_file   = base_fp+"../data/Coco/Annotations/captions_val2014.json"
		train_file = base_fp+"../data/Coco/Annotations/captions_train2014.json" 
		tiny_file = base_fp+"../data/Coco/Annotations/captions_tiny2014.json" 

		self.data_source = {train_file:"train",val_file:"val",tiny_file:"tiny"}

		data_options = {0:[train_file],1:[val_file],2:[train_file,val_file],3:[tiny_file]}

		self.save_loc = base_fp+"saved_items/CocoCaptions_saved_%i.pkl"%data
		self.data = self.create_data(data_options[data]) #{(image_id,source):[caption1,caption2,....]}

		self.image_locations = {"train":base_fp+"../data/Coco/Train2014/COCO_train2014_",
							    "val":base_fp+"../data/Coco/Val2014/COCO_val2014_",
							    "tiny":base_fp+"../data/Coco/Tiny2014/COCO_train2014_",
							   }
		self.resnet_locations ={"train":base_fp+"../data/Coco_ResNet/Train2014/COCO_train2014_",
							    "val":base_fp+"../data/Coco_ResNet/Val2014/COCO_val2014_",
							    "tiny":base_fp+"../data/Coco_ResNet/Tiny2014/COCO_train2014_",
							   }

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
		return self.data

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




if __name__ == "__main__":
	
	'''
	CG = CaptionGloveVectors()
	G = GloVeVectors()
	'''
	Captions = CocoCaptions(3)
	Captions.get_image((366897,"tiny"))

