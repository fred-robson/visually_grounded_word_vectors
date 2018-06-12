import skipthoughts, pickle
import eval_trec, eval_rank
import numpy as np
import argparse



def load_inputs(path):
	'''
	Returns captions,resnet_embeddings,vectors
	'''

	# load captions and resnet embeddings
	split = pickle.load(open(path, "rb" ))
	ret = None

	if len(split[0])==2:
		# encode captions via skipthought, if there are not emebddings already include
		model = skipthoughts.load_model()
		encoder = skipthoughts.Encoder(model)
		captions, resnet_embeddings = [s[0] for s in split],[s[1].flatten() for s in split]
		vectors = encoder.encode(captions)
		ret = captions,resnet_embeddings,vectors
	elif len(split[0])==3:
		captions, resnet_embeddings,vectors = [s[0] for s in split],[s[1].flatten() for s in split], [s[2].flatten() for s in split]
		vectors = np.array(vectors)
		ret = captions,resnet_embeddings,vectors
	else: 
		raise "Input pickled vectors should be a list of two tuples (caption,resnet) or\
		three-tuples (caption,resnet,embedding)"

	return ret






if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='path to pickled input', default='coco_split.p')
	parser.add_argument('--path2', help='path to pickled input', default='coco_split.p')
	parser.add_argument('--concat',action="store_true")
	args = parser.parse_args()

	print("\n> =============================================================<")
	print("> Yo make sure to run: THEANO_FLAGS='floatX=float32' python st.py <")
	print("\n> =============================================================<")


	captions,resnet_embeddings,vectors = load_inputs(args.path)	
	if args.concat:
		print "Concatenating Vectors"
		captions2,_,vectors2 = load_inputs(args.path2)
		new_vectors = []
		for c1,v1,c2,v2 in zip(captions,vectors,captions2,vectors2):
			print c1,c2
			concat_v = np.concatenate((v1,v2))
			new_vectors.append(concat_v)
	
			vectors = np.array(new_vectors)
			





	# testing types
	print "Types:",type(vectors),type(vectors[0]),type(resnet_embeddings),type(resnet_embeddings[0])

	# shape
	print "Resnet Shapes",resnet_embeddings[0].shape
	print "Vectors Shape:",vectors.shape

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
	eval_rank.trainer(train, dev,dim_im=1000,dim_s=vectors.shape[1],saveto=saveto,validFreq=10)
	eval_rank.evaluate(test, saveto, evaluate=True)

	#eval_trec.evaluate(encoder, evalcv=False, evaltest=True)