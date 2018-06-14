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


def main(w,i):
	G,CG,vocab_intersection = initial_filtering()	
	print("GloVe Vectors")
	print(G.nearest_neighbors(w,i))
	print("OurGlove")
	print(CG.nearest_neighbors(w,i))


if __name__ == "__main__":
	print(sys.argv)
	main(sys.argv[1],int(sys.argv[2]))