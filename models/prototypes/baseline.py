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
<<<<<<< HEAD
from keras.layers import Masking
>>>>>>> e4af9544b71e260081ee55e72d983efee44448c8
from tensorflow.contrib.training import HParams
import keras.backend as K


class Cap2(object):

	def __init__(self, h_params, embeddings=None,model_type='cap2all'):
<<<<<<< HEAD
		if type(h_params) is HParams:
			self.h_params = h_params
		else:
			self.h_params = HParams(**h_params)
>>>>>>> e4af9544b71e260081ee55e72d983efee44448c8
		self.embedding_matrix = embeddings
		self.model_type = model_type
		self.model = self.build()

	def build(self):
		model=None
<<<<<<< HEAD
		inputs=[]
		outputs=[]
		loss={}

		## Encoder ##
		encoder_input = Input(shape=(self.h_params.max_seq_length,),dtype='int32', name='encoder_input')
		x = Masking(mask_value=0)(encoder_input)
		x = Embedding(self.h_params.num_embeddings, self.h_params.embed_dim, weights=[self.embedding_matrix], 
						input_length=self.h_params.max_seq_length, 
						trainable=False )(x)
		x = Dense(self.h_params.embed_dim)(x)
		f_out, b_out, f_state_h, f_state_c, b_state_h, b_state_c = Bidirectional(LSTM(self.h_params.hidden_dim, 
																				dropout=self.h_params.dropout, 
																				recurrent_dropout=self.h_params.dropout, 
																				return_state=True), merge_mode=None)(x)

		encoder_out = Maximum()([f_state_h, b_state_h])
		encoder_out_cell = Maximum()([f_state_c, b_state_c])

		inputs += [encoder_input]

		if self.model_type == 'cap2img' or self.model_type == 'cap2all':
			img_x = Dense(self.h_params.output_dim,activation=self.h_params.activation)
			outputs += [img_output]

		if self.model_type == 'cap2cap' or self.model_type == 'cap2all':
			decoder_input = Input(shape=(self.h_params.max_seq_length,),dtype='int32', name='decoder_input')
			x_dec = Masking(mask_value=0)(decoder_input)
			x_dec = Embedding(self.h_params.num_embeddings, self.h_params.embed_dim, weights=[self.embedding_matrix], 
							input_length=self.h_params.max_seq_length, 
							trainable=False )(x_dec)
			x_dec = Dense(self.h_params.embed_dim)(x_dec)
			x_dec = LSTM(self.h_params.hidden_dim, dropout=self.h_params.dropout, 
							recurrent_dropout=self.h_params.dropout, 
							return_sequences=True)(x_dec, initial_state=[encoder_out, encoder_out_cell])

			decoder_output = Dense(self.h_params.num_embeddings, activation='softmax', name='decoder_output')(x_dec)

			inputs += [decoder_input]
			outputs += [decoder_output]
			if self.model_type == 'cap2cap':
				loss['decoder_output'] = 'sparse_categorical_crossentropy'
>>>>>>> e4af9544b71e260081ee55e72d983efee44448c8

		model = Model(inputs=inputs, outputs=outputs)
		
		model.name = self.model_type
<<<<<<< HEAD
		model.compile(loss=loss,optimizer=self.h_params.optimizer,metrics=['sparse_categorical_accuracy'])
>>>>>>> e4af9544b71e260081ee55e72d983efee44448c8
		return model

