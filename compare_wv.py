from utils.data_utils import *
from utils.word_vec_utils import *
import sys


def initial_filtering():
	CG = OurGloveVectors()
	print(len(CG.i2w))
	G = GloVeVectors()
	vocab_intersection = set(G.get_vocab()) & set(CG.get_vocab())
	CG.filter_wv(vocab_intersection)
	G.filter_wv(vocab_intersection)
	return G,CG,vocab_intersection


def main(w):
	G,CG,vocab_intersection = initial_filtering()	
	print("GloVe Vectors")
	print(G.nearest_neighbors(w,5))
	print("OurGlove")
	print(CG.nearest_neighbors(w,5))


if __name__ == "__main__":
	print(sys.argv)
	main(sys.argv[1])