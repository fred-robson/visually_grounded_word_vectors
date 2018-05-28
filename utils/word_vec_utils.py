'''
Folder for word embeddings utils
'''
from scipy.spatial.distance import cosine
import os
import numpy as np
from tqdm import tqdm
from keras.layers import Embedding


base_fp = os.path.dirname(os.path.abspath(__file__))+"/"
unk = "<unk>"

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

		if not unk in self.w2i:
			np.append(self.vectors,np.zeros((dimensions)))
			self.w2i[unk] = len(self.i2w)
			self.i2w.append(unk)

	def get_vector(self,word):
		#Retunrs the vector for a particular @word
		return self.vectors[self.w2i[word]]

	def words_to_vectors(self,words):
		#converts a list of words into an np array 
		output = []
		for w in words:
			output.append(self.get_vector(w))
		return np.array(output)

	def words_to_indices(self,words):
		output = []
		for w in words:
			if w in self.w2i:
				output.append(self.w2i[w])
			else: 
				output.append(self.w2i[unk])
		return np.array(output)

	def indices_to_words(self,indices):
		return [self.i2w[i] for i in indices]

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

	def get_embedding_matrix(self):
		return self.vectors


def load_txt_vectors(filepath,dimensions):
	'''
	Loads a series of vectors from a txt file
	'''	
	num_words = sum(1 for line in open(filepath))
	words = []
	vectors = np.zeros((num_words,dimensions))
	print(filepath)
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
	X = CaptionGloveVectors()
	print(X.words_to_vectors(["man","woman"]).shape)




