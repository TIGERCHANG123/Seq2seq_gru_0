from __future__ import print_function
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, GRU
import tensorflow as tf

class Encoder(tf.keras.Model):
  def __init__(self, num_encoder_tokens, latent_dim, embedding_dim=128, batch_sz=128):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = latent_dim
    self.embedding = tf.keras.layers.Embedding(num_encoder_tokens, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x):
    x=tf.math.argmax(x, axis=-1)
    x=self.embedding(x)
    output, state = self.gru(x)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
  def __init__(self, num_decoder_tokense, latent_dim, embedding_dim=128,  batch_sz=128):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = latent_dim
    self.embedding = tf.keras.layers.Embedding(num_decoder_tokense, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(num_decoder_tokense)

  def call(self, x, hidden):
    x=tf.math.argmax(x, axis=-1)
    x = self.embedding(x)
    output, state = self.gru(x, initial_state=hidden)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.fc(output)
    return x, state



