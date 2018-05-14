from utils.data_utils import *
import sys


def initial_filtering():
	CG = CaptionGloveVectors()
	G = GloVeVectors()
	vocab_intersection = set(G.get_vocab()) & set(CG.get_vocab())
	CG.filter_wv(vocab_intersection)
	G.filter_wv(vocab_intersection)
	return G,CG,vocab_intersection


def main():
	G,CG,vocab_intersection = initial_filtering()	
	w = "eats"
	print(G.nearest_neighbors(w,5))
	print(CG.nearest_neighbors(w,5))


if __name__ == "__main__":
	
	main()