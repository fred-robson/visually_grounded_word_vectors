import keras
import tensorflow as tf
import numpy as np 
from keras.models import Model 
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers import Maximum
from keras.layers import Masking
from keras.layers import Layer
from keras.layers import Lambda 
from keras.layers import Add
from keras.layers import Multiply
from tensorflow.contrib.training import HParams
import keras.backend as K
from keras.utils import plot_model
from keras import metrics as k_metrics

def KL_divergence(y_true, y_pred):
	loss = 0.0
	return loss

def ranking_loss(y_true, y_pred):
	loss = K.constant(0)
	return loss

class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs

class Cap2(object):

	def __init__(self, h_params, embeddings=None, graph_path='./models/visualization/{0}.png'):
		if type(h_params) is HParams:
			self.h_params = h_params
		else:
			self.h_params = HParams(**h_params)
		self.embedding_matrix = embeddings
		self.model_type = None
		self.graph_path = graph_path
		self.model = None

	def visualize(self):
		path = self.graph_path.format(self.model_type)
		plot_model(self.model, to_file=path)

	def _encoder(self):
		## Encoder ##
		encoder_input = Input(shape=(self.h_params.max_seq_length,),dtype='int32', name='encoder_input')
		#x = Masking(mask_value=0)(encoder_input)
		glove_embedding_encoder = Embedding(self.h_params.num_embeddings, self.h_params.embed_dim, weights=[self.embedding_matrix], 
						input_length=self.h_params.max_seq_length, 
						trainable=False , mask_zero=True, name='GloVe_embedding_encoder')
		x = glove_embedding_encoder(encoder_input)
		x = Dense(self.h_params.embed_dim)(x)
		f_out, b_out, f_state_h, f_state_c, b_state_h, b_state_c = Bidirectional(LSTM(self.h_params.hidden_dim, 
																				dropout=self.h_params.dropout, 
																				recurrent_dropout=self.h_params.dropout, 
																				return_state=True), merge_mode=None, name='encoder')(x)
		encoder_out = Maximum(name='encoder_output')([f_state_h, b_state_h])
		encoder_out_cell = Maximum()([f_state_c, b_state_c])
		return encoder_input, encoder_out, encoder_out_cell

	def _build(self):
		return None, None

	def load_model(self, path):
		self.model = keras.models.load_model(path)

	def compile(self):
		inputs, outputs = self._build()
		model = Model(inputs=inputs, outputs=outputs)
		model.name = self.model_type
		model.compile(loss=self.loss,loss_weights=self.loss_weights,optimizer=self.h_params.optimizer,metrics=self.metrics)
		self.model = model

class Cap2Cap(Cap2):
	loss = {'decoder_output':'sparse_categorical_crossentropy'}
	metrics = ['sparse_categorical_accuracy']
	loss_weights=None
	def __init__(self, h_params, **kwds):
		super().__init__(h_params, **kwds)
		model_type = 'cap2cap'

	def _decoder(self, initial_state):
		decoder_input = Input(shape=(self.h_params.max_seq_length,),dtype='int32', name='decoder_input')
		#x_dec = Masking(mask_value=0)(decoder_input)
		glove_embedding_decoder = Embedding(self.h_params.num_embeddings, self.h_params.embed_dim, weights=[self.embedding_matrix], 
					input_length=self.h_params.max_seq_length, 
					trainable=False , mask_zero=True, name='GloVe_embedding_decoder')
		x_dec = glove_embedding_decoder(decoder_input)
		x_dec = Dense(self.h_params.embed_dim)(x_dec)
		x_dec = LSTM(self.h_params.hidden_dim, dropout=self.h_params.dropout, 
						recurrent_dropout=self.h_params.dropout, 
						return_sequences=True, name='decoder')(x_dec, initial_state=initial_state)

		decoder_output = Dense(self.h_params.num_embeddings, activation='softmax', name='decoder_output')(x_dec)
		return decoder_input, decoder_output

	def _build(self):
		encoder_input, encoder_output, encoder_out_cell = self._encoder()
		decoder_input, decoder_output = self._decoder([encoder_output, encoder_out_cell])
		inputs = [encoder_input, decoder_input]
		outputs = [decoder_output]
		return inputs, outputs

class Cap2Img(Cap2):
	loss = {'projection_output': 'mean_squared_error'}
	loss_weights=None
	metrics = [k_metrics.mse]
	def __init__(self, h_params, **kwds):
		super().__init__(h_params, **kwds)
		self.model_type = 'cap2img'

	def _projection(self, x):
		projection_output = Dense(self.h_params.latent_dim,activation=self.h_params.activation, name='projection_output')(x)
		return projection_output

	def _build(self):
		encoder_input, encoder_output, encoder_out_cell = self._encoder()
		inputs = [encoder_input]
		projection_output = self._projection(encoder_output)
		outputs = [projection_output]
		return inputs, outputs



class Cap2All(Cap2Cap, Cap2Img):
	loss = {**Cap2Cap.loss, **Cap2Img.loss}
	metrics = {'decoder_output':'sparse_categorical_accuracy'}
	loss_weights={'decoder_output':1., 'projection_output':0.5}
	def __init__(self, h_params, **kwds):
		super().__init__(h_params, **kwds)
		self.model_type = 'cap2all'
		#print(self.loss)

	def _build(self):
		encoder_input, encoder_output, encoder_out_cell = self._encoder()
		projection_output = self._projection(encoder_output)
		decoder_input, decoder_output = self._decoder([encoder_output, encoder_out_cell])
		inputs = [encoder_input, decoder_input]
		outputs = [projection_output, decoder_output]
		return inputs, outputs

class Vae2All(Cap2Cap, Cap2Img):
	loss = {**Cap2Cap.loss, **Cap2Img.loss}
	metrics = {'decoder_output':'sparse_categorical_accuracy'}
	loss_weights={'decoder_output':1., 'projection_output':0.5}
	def __init__(self, h_params, **kwds):
		super().__init__(h_params, **kwds)
		self.model_type = 'vae2all'
		#print(self.loss)
	
	def _variational(self, x):
		z_mu = Dense(self.h_params.hidden_dim)(x)
		z_log_var = Dense(self.h_params.hidden_dim)(x)

		z_mu, z_log_var = KLDivergenceLayer(name='KL_divergence')([z_mu, z_log_var])
		z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
		eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0],
		                                          self.h_params.hidden_dim)))
		z_eps = Multiply()([z_sigma, eps])
		z = Add()([z_mu, z_eps])
		return z, eps

	def _build(self):
		encoder_input, encoder_output, encoder_out_cell = self._encoder()
		z, eps = self._variational(encoder_output)
		projection_output = self._projection(z)
		decoder_input, decoder_output = self._decoder([z, encoder_out_cell])
		inputs = [encoder_input, decoder_input, eps]
		outputs = [projection_output, decoder_output]
		return inputs, outputs


def get_model(name):
	model_dict = {'cap2cap':Cap2Cap, 'cap2img':Cap2Img, 'cap2all':Cap2All, 'vae2all':Vae2All}
	try:
		model = model_dict[name]
	except:
		print("the model \'"+name+"\' doesn't exist")
		model = None
	return model

if __name__ == '__main__':
	embedding_matrix = np.random.randn(10000,50)
	hparams = HParams(learning_rate=0.01, hidden_dim=1024,
						optimizer='adam', dropout= 0.5, 
						max_seq_length=50,
						embed_dim=embedding_matrix.shape[-1],
						num_embeddings=embedding_matrix.shape[0],
						activation='relu',
						latent_dim=1000)
	cap2 = Vae2All(hparams, embeddings=embedding_matrix)
	cap2.compile()
	cap2.visualize()

