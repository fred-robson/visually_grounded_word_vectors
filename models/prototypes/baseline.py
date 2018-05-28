import keras
import tensorflow as tf
import numpy as np 
from keras.models import Model 
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Embedding
from tensorflow.contrib.training import HParams


class Cap2(object):

	def __init__(self, h_params, embeddings=None,model_type='cap2all'):
		self.h_params = HParams(**h_params)
		self.embedding_matrix = embeddings
		self.model_type = model_type
		self.model = self.build()

	def _encoder(self):
		model = Sequential()
		model.add
		model.add(Bidirectional(LSTM(hidden_dim, dropout=self.h_params.dropout, recurrent_dropout=self.h_params.dropout, return_sequences=True)))
		model.add(TimeDistributed(Dense()))
		return model

	def _decoder(self, model):

	def _map2latent(self, model):

	def fit(self, x, labels):
		if self.model_type == 'cap2all':
			assert (len(labels) == 2),"Not Enough Labels For Cap2All"

	def build(self):
		model = self._encoder()
		if self.model_type == 'cap2cap':
			model = self._decoder(model)
		if self.mode_type == 'cap2img':
			model = self._map2latent(model)
		model.name = self.model_type
		model.compile()
		return model




def get_encoder(hidden_dim, model='all', hidden_dim=20, encoder_dropout=0.0, encoder_recurrent_dropout=0.0):
	
	model = Sequential()
	model.add(Bidirectional(LSTM(hidden_dim, dropout=encoder_dropout, recurrent_dropout=encoder_dropout, return_sequences=True)))
	model.add(TimeDistributed(Dense()))

	return model

def get_decoder

def cap_2_all(data, hidden_dim=20, encoder_dropout=0.0, encoder_recurrent_dropout=0.0):
	input_cap, label_cap, label_img = data
def cap_2_cap(data, hidden_dim=20, encoder_dropout=0.0, encoder_recurrent_dropout=0.0):
	input_cap, label_cap = data
	model = encoder(hidden_dim, dropout=encoder_dropout, recurrent_dropout=encoder_dropout, return_sequences=True)
def cap_2_img(data, hidden_dim=20, encoder_dropout=0.0, encoder_recurrent_dropout=0.0):
	input_cap, label_img = data
	model = Sequential()
	model.add(Bidirectional(LSTM(hidden_dim, dropout=encoder_dropout, recurrent_dropout=encoder_dropout, return_sequences=True)))
	model.add(TimeDistributed(Dense()))

