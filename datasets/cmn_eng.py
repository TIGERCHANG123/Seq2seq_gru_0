from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
data_path = '/home/tigerc/datasets'

class cmn_eng_dataset():
    def __init__(self, file_path=data_path+'/cmn_eng/cmn.txt'):
        self.latent_dim = 256  # Latent dimensionality of the encoding space.
        self.num_samples = 10000  # Number of samples to train on.

        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        self.num_samples = min(self.num_samples, len(lines) - 1)
        for line in lines[: self.num_samples]:
            input_text, target_text, _ = line.split('\t')
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        self.input_texts = input_texts
        self.target_texts = target_texts

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length

        print('Number of samples:', len(input_texts))
        print('Number of unique input tokens:', num_encoder_tokens)
        print('Number of unique output tokens:', num_decoder_tokens)
        print('Max sequence length for inputs:', max_encoder_seq_length)
        print('Max sequence length for outputs:', max_decoder_seq_length)

        input_token_index = dict(
            [(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])

        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.
            decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
            decoder_target_data[i, t:, target_token_index[' ']] = 1.
        self.encoder_input_data = encoder_input_data
        self.decoder_input_data = decoder_input_data
        self.decoder_target_data = decoder_target_data
    def parse(self, x1, x2, y):
        return (x1, x2), y

    def generator(self):
        l = list(range(self.num_samples))
        random.shuffle(l)
        for i in l:
            yield ((self.encoder_input_data[i], self.decoder_input_data[i]), self.decoder_target_data[i])
    def train_dataset(self):
        ds = tf.data.Dataset.from_generator(self.generator, output_types=((tf.float32, tf.float32), tf.float32))
        # ds = ds.shuffle(self.num_samples)
        # ds = ds.map(self.parse)
        # dt = ds.take(int(self.num_samples * 0.8)).batch(128)
        # dv = ds.skip(int(self.num_samples * 0.2)).batch(128)
        dt = ds.batch(128)
        return dt
    def decode_sequence_GRU_Attention(self,models, input_seq):
        encoder_model = models[0]
        decoder_model = models[1]

        reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

        # Encode the input as state vectors.
        output_value, states_value = encoder_model(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, states_value = decoder_model(target_seq, states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[-1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
               len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
        return decoded_sentence

    def test(self, models, txt):
        for seq_index in range(100):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = self.encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = self.decode_sequence_GRU_Attention(models, input_seq)
            print('-', file=txt)
            print('Input sentence:', self.input_texts[seq_index], file=txt)
            print('Decoded sentence:', decoded_sentence, file=txt)

