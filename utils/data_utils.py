#Classes and functions for accessing stuff from data files
import json, os
from collections import defaultdict
from nltk.tokenize import word_tokenize as tokenize
from scipy.spatial.distance import cosine

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
			3 = tiny
		'''
		val_file   = "../data/Coco/Annotations/captions_val2014.json"
		train_file = "../data/Coco/Annotations/captions_train2014.json" 
		tiny_file = "../data/Coco/Annotations/captions_tiny2014.json" 

		data_options = {0:[train_file],1:[val_file],2:[train_file,val_file],3:[tiny_file]}

		self.save_loc = "../data/Coco/Annotations/CocoCaptions_saved_%i.pkl"%data
		self.data = self.create_data(data_options[data]) #{image_id:[caption1,caption2,....]}


	def create_data(self,file_names):
		'''
		Loads data in the format: {image_id:[caption1,caption2,....]}
		'''
		if os.path.isfile(self.save_loc):
			return pkl.load(open(self.save_loc,'rb')) 

		loaded_jsons = []
		for f in file_names: 
			loaded_jsons+=self.load_json(f)

		data = defaultdict(lambda: [])
		print("Running one-time tokenization of captions")
		for v in tqdm(loaded_jsons):
			data[v['image_id']].append(tokenize(v['caption'].lower()))

		pkl.dump(dict(data),open(self.save_loc,"wb"))

		return data

	def load_json(self,file_name):
		with open(file_name) as f:
			data = json.load(f)
		return data['annotations']

	def get_vocab(self):
		vocab = set()
		for v in self.data:
			vocab.add(v)
		return vocab

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




class WordVectors():
	'''
	A general class for manipulating word vectors. It is assumed that WordVectors
	will never be initialzed except as a parent class. For example see, GLoVe vectors
	below
	'''
	
	def __init__(self,words,vectors):
		self.i2w = words #list(words)
		self.vectors = vectors #np.arrray(), shape=(num_words,nun_dimensions)
		self.w2i = {w:i for i,w in enumerate(words)}

	def get_vector(self,word):
		#Retunrs the vector for a particular @word
		return self.vectors[self.w2i[word]]

	def nearest_neighbors(self,word,k=None,dist_func=cosine):
		'''
		Finds the nearest k neighbors to @word as determined by dist_func. 
		@args: 
			- word: word to find nearest neighbors to (str)
			- k: num neighbors (int). If none, then returns all
			- dist_func: function(x,y) where x and y are both 1d arrays
		@return: 
			- list of (nearest neighbor words, distance)
		'''
		if k == None: k = len(self.i2w)-1 #-1 to remove self
		wv = self.get_vector(word)
		func = lambda x: dist_func(x,wv)
		distances = np.apply_along_axis(func,1,self.vectors)
		ordered_indices = np.argsort(distances)
		return [(self.i2w[i],distances[i]) for i in ordered_indices[1:k+1]]

	def get_vocab(self):
		return self.i2w 

	def filter_wv(self,words):
		#Removes any word vector that is not part of @words
		new_i2w = list(words)
		new_vectors = np.zeros(shape=(len(new_i2w),len(self.vectors[0])))
		for i,w in enumerate(new_i2w):
			new_vectors[i] = self.get_vector(w)
		self.i2w = new_i2w
		self.vectors = new_vectors
		self.w2i = {w:i for i,w in enumerate(new_i2w)}


def load_txt_vectors(filepath,dimensions):
	'''
	Loads a series of vectors from a txt file
	'''	
	num_words = sum(1 for line in open(filepath))
	words = []
	vectors = np.zeros((num_words,dimensions))
	with open(filepath) as f:
		for i,line in tqdm(enumerate(f),total=num_words,desc="Reading %d dimensional vectors"%dimensions):
			row = line.split()
			word = row[0]
			vector = np.array(row[1:])

			words.append(word)
			vectors[i] = vector

	return words,vectors


class GloVeVectors(WordVectors):

	def __init__(self,dimensions=50):
		if not dimensions in {50,100,200,300}:
			raise ValueError("GloVe Dimension does not exist")
		filepath = base_fp+"../data/Vectors/GloVe/glove.6B.%dd.txt"%dimensions
		self.dimensions = dimensions
		words,vectors = load_txt_vectors(filepath,dimensions)
		WordVectors.__init__(self,words,vectors)


class CaptionGloveVectors(WordVectors):

	def __init__(self,dimensions=50):
		if not dimensions in {50}: #Update as more dimensions added 
			raise ValueError("GloVe Dimension does not exist")
		filepath = base_fp+"../data/Vectors/VisuallyGrounded/vectors%d.txt"%dimensions
		self.dimensions = dimensions
		words,vectors = load_txt_vectors(filepath,dimensions)
		WordVectors.__init__(self,words,vectors)

	

	


	



if __name__ == "__main__":
	
	'''
	CG = CaptionGloveVectors()
	G = GloVeVectors()
	'''
	Captions = CocoCaptions(0)
	Captions.build_corpus()
