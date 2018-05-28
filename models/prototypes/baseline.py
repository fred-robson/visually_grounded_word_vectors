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
from tensorflow.contrib.training import HParams
import keras.backend as K


class Cap2(object):

	def __init__(self, h_params, embeddings=None,model_type='cap2all'):
		self.h_params = HParams(**h_params)
		self.embedding_matrix = embeddings
		self.model_type = model_type
		self.model = self.build()

	def build(self):
		model=None

		## Encoder ##
		encoder_input = Input(shape=(h_params.max_seq_length,), dtype='int32', name='encoder_input')
		x = Embedding(input_dim, output_dim, weights=[embedding_matrix], input_length=h_params.max_seq_length, trainable=False )(main_input)
		x = Dense(h_params.embed_dim, input_shape())
		x = Bidirectional(LSTM(hidden_dim, dropout=self.h_params.dropout, recurrent_dropout=self.h_params.dropout, return_sequences=True), merge_mode=None)(x)
		x = Maximum()(x)

		if self.model_type == 'cap2cap' or self.model_type == 'cap2all':
			text_x = Bidirectional(LSTM(hidden_dim, dropout=self.h_params.dropout, recurrent_dropout=self.h_params.dropout, return_sequences=True))(x)
			text_output = TimeDistributed(Dense())(text_x)
			if self.model_type == 'cap2cap':
				model = Model(inputs=[encoder_input,decoder_input], outputs=[decoder_output])
		if self.mode_type == 'cap2img':
			img_x = Dense(h_params.output_dim,activation=h_params.activation, input_shape=())
		if self.model_type == 'cap2all':
			model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output, img_output])
		
		model.name = self.model_type
		model.compile(loss='sparse_categorical_crossentropy',optimizer=h_params.optimizer,metrics=['sparse_categorical_accuracy'])
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

