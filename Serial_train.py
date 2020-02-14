# -*- coding:utf-8 -*-
import  os
import  tensorflow as tf
from    tensorflow.keras import optimizers
from models.seq2seq_gru import Encoder, Decoder
from datasets.cmn_eng import cmn_eng_dataset
from show_pic import draw
from tensorflow import keras

model_dataset = 'seq2seq_gru_cmn_eng'
root = '/home/tigerc/temp'

# @tf.function
# def compute_loss(pred, real):
#     return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(real, pred))
@tf.function
def compute_loss(real, pred):
    real = tf.math.argmax(real, axis=-1)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def train_one_step(models, optimizer, x, y):
    encoder = models[0]
    decoder = models[1]
    with tf.GradientTape() as tape:
        # watch will make these tensors traced by gradient
        tape.watch(encoder.trainable_variables)
        tape.watch(decoder.trainable_variables)
        encoder_output, encoder_state = encoder(x)
        decoder_input = tf.expand_dims(y[:, 0], 1)
        decoder_state = encoder_state
        loss=0
        for t in range(1, y.shape[1]):
            predictions, decoder_state = decoder(decoder_input, decoder_state)
            loss += compute_loss(y[:, t], predictions)
            decoder_input = tf.expand_dims(y[:, t], 1)
    variables = encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(loss, variables)
    # print('loss: ', loss/(y.shape[1]-1), 'grads: ', grads)
    optimizer.apply_gradients(zip(grads, variables))
    return loss.numpy()/(y.shape[1]-1)

def train(epoch,  pic, models, dataset, optimizer):
  train_ds = dataset.train_dataset()
  loss = 0.0
  accuracy = 0.0
  for step, ((x1, x2), y) in enumerate(train_ds):
    loss = train_one_step(models, optimizer, x1, x2)
    pic.add([loss, accuracy])
    pic.save(root + '/temp_pic_save/'+model_dataset)
    if step%500==0:
      # loss, accuracy = test(model, train_dv)
      print('epoch', epoch, ': loss', loss, '; accuracy', accuracy)
  print('epoch', epoch, ': loss', loss, '; accuracy', accuracy)

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    if not (os.path.exists(root + '/temp_pic/' + model_dataset)):
        os.makedirs(root + '/temp_pic/' + model_dataset)
    if not (os.path.exists(root + '/temp_model_save/' + model_dataset)):
        os.makedirs(root + '/temp_model_save/' + model_dataset)
    if not (os.path.exists(root + '/temp_pic_save/' + model_dataset)):
        os.makedirs(root + '/temp_pic_save/' + model_dataset)
    if not(os.path.exists(root + '/temp_txt_save/'+model_dataset)):
        os.makedirs(root + '/temp_txt_save/'+model_dataset)

    pic = draw(10)
    if not(os.path.exists(root+'/temp_txt_save/'+model_dataset+'/validation.txt')):
        txt = open(root+'/temp_txt_save/'+model_dataset+'/validation.txt','w')
    else:
        txt = open(root+'/temp_txt_save/'+model_dataset+'/validation.txt', 'a')
    dataset = cmn_eng_dataset()

    encoder = Encoder(num_encoder_tokens=dataset.num_encoder_tokens, embedding_dim=128, latent_dim=dataset.latent_dim)
    decoder = Decoder(num_decoder_tokense=dataset.num_decoder_tokens, embedding_dim=128, latent_dim=dataset.latent_dim)
    optimizer = optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    checkpoint_dir = root + '/temp_model_save/' + model_dataset
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # if os.path.isdir(checkpoint_dir):
    #    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    for epoch in range(10):
        train(models=[encoder, decoder], dataset=dataset, epoch=epoch,  pic=pic, optimizer=optimizer)
        pic.show(root+'/temp_pic/' + model_dataset + '/pic')
        checkpoint.save(file_prefix=checkpoint_prefix)
    dataset.test([encoder, decoder], txt)
    txt.close()
    return

if __name__ == '__main__':
    main()