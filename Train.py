# -*- coding:utf-8 -*-
import os
import tensorflow as tf

ubuntu_root='/home/tigerc/temp'
windows_root='D:/Automatic/SRTP/GAN/temp'
model_dataset = 'seq2seq_gru_cmn_eng'
root = ubuntu_root

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

class train_one_epoch():
    def __init__(self, encoder, decoder, train_dataset, optimizer, metrics):
        self.encoder=encoder
        self.decoder=decoder
        self.optimizer = optimizer
        self.train_loss, self.train_accuracy= metrics
        self.train_dataset = train_dataset
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ])
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            # watch will make these tensors traced by gradient
            tape.watch(self.encoder.trainable_variables)
            tape.watch(self.decoder.trainable_variables)
            encoder_output, encoder_state = self.encoder(x)
            decoder_input = tf.expand_dims(y[:, 0], 1)
            decoder_state = encoder_state
            loss = 0
            for t in range(1, y.shape[1]):
                predictions, decoder_state = self.decoder(decoder_input, decoder_state)
                loss += loss_function(y[:, t], predictions)
                decoder_input = tf.expand_dims(y[:, t], 1)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        self.train_loss(loss)
        self.train_accuracy(y, predictions)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ])

    def train(self, epoch,  pic):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(self.train_dataset):
            self.train_step(inp=inp, tar=tar)
            pic.add([self.train_loss.result().numpy(), self.train_accuracy.result().numpy()])
            pic.save(root + '/temp_pic_save/' + model_dataset)
        print('epoch: {}, loss: {}, accuracy: {}'.format(epoch, self.train_loss.result(), self.train_accuracy.result()))