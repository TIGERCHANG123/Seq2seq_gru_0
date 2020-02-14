# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from models.seq2seq_gru import Encoder, Decoder
from datasets.cmn_eng import cmn_eng_dataset
from show_pic import draw
from Train import train_one_epoch
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

ubuntu_root='/home/tigerc/temp'
windows_root='D:/Automatic/SRTP/GAN/temp'
model_dataset = 'seq2seq_gru_1_cmn_eng'
root = ubuntu_root

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    if not (os.path.exists(root + '/temp_pic/' + model_dataset)):
        os.makedirs(root + '/temp_pic/' + model_dataset)
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
    train_dataset = dataset.train_dataset()

    optimizer = tf.keras.optimizers.Adam()

    encoder = Encoder(num_encoder_tokens=dataset.num_encoder_tokens, embedding_dim=128, latent_dim=dataset.latent_dim)
    decoder = Decoder(num_decoder_tokense=dataset.num_decoder_tokens, embedding_dim=128, latent_dim=dataset.latent_dim)

    checkpoint_path = root + '/temp_model_save/' + model_dataset
    ckpt = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    train = train_one_epoch(encoder=encoder, decoder=decoder, train_dataset=train_dataset, optimizer=optimizer, metrics=[train_loss, train_accuracy])
    print('start training')
    for epoch in range(10):
        train.train(epoch=epoch, pic=pic)
        pic.show(root+'/temp_pic/' + model_dataset + '/pic')
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
    txt.close()
    return

if __name__ == '__main__':
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    main()

