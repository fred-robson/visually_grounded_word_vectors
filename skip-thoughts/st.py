import skipthoughts, pickle
import eval_trec, eval_rank
import numpy as np
import argparse



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='path to pickled input', default='coco_split.p')
	args = parser.parse_args()

	print("\n> =============================================================<")
	print("> Yo make sure to run: THEANO_FLAGS='floatX=float32' python st.py <")
	print("\n> =============================================================<")
	# load captions and resnet embeddings
	split = pickle.load(open(args.path, "rb" ) )
	
	#separate captions and resnet embeddings
	

	# encode captions via skipthought, if there are not emebddings already included
	
	captions, resnet_embeddings,vectors = None,None,None

	if len(split[0])==2:
		captions, resnet_embeddings = [s[0] for s in split],[s[1].flatten() for s in split]
		vectors = encoder.encode(captions)
	elif len(split[0])==3:
		captions, resnet_embeddings,vectors = [s[0] for s in split],[s[1].flatten() for s in split], [s[2].flatten for s in split]
	else: 
		raise "Input pickled vectors should be a list of two tuples (caption,resnet) or\
		three-tuples (caption,resnet,embedding)"

	model = skipthoughts.load_model()
	encoder = skipthoughts.Encoder(model)
	

	# testing types
	print(type(vectors),type(vectors[0]),type(resnet_embeddings),type(resnet_embeddings[0]))

	# shape
	print(resnet_embeddings[0].shape)

	# generate train, dev, and test set
	print(len(vectors),len(captions),len(resnet_embeddings))
	train_size = int((len(vectors)*0.7)//1)
	dev_size = int((len(vectors)*0.1)//1)
	test_size = int((len(vectors)*0.2)//1)

	print(train_size,dev_size,test_size)

	train = [captions[:train_size],np.asarray(resnet_embeddings[:train_size]),vectors[:train_size]]
	dev = [captions[train_size:train_size+dev_size],np.asarray(resnet_embeddings[train_size:train_size+dev_size]),vectors[train_size:train_size+dev_size]]
	test = [captions[:-test_size],np.asarray(resnet_embeddings[:-test_size]),vectors[:-test_size]]
	#print(train)
	saveto = "mod.npz"
	eval_rank.trainer(train, dev,dim_im=1000,saveto=saveto,validFreq=10)
	eval_rank.evaluate(test, saveto, evaluate=True)

	#eval_trec.evaluate(encoder, evalcv=False, evaltest=True)