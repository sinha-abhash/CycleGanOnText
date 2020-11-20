import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, LayerNormalization, Conv1D, LeakyReLU, BatchNormalization, Dropout, Flatten, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np

import logging

logging.basicConfig(level=logging.INFO)


class Discriminator:
    def __init__(self, latent_dim, max_len_question):
        self.latent_dim = latent_dim
        self.max_len_question = max_len_question

    def get_model(self):
        input_text = Input(shape=(self.max_len_question, self.latent_dim))
        mask = Masking(mask_value=np.zeros(self.latent_dim))(input_text)
        """
        # lstm
        lstm = LSTM(self.latent_dim, return_sequences=False, return_state=False)
        lstm_output = lstm(input_vec)
        layer_norm1 = LayerNormalization(axis=1)(lstm_output)

        # dense
        dense = Dense(1, activation='sigmoid')
        dense_output = dense(layer_norm1)
        """

        fe = Conv1D(16, 3, strides=2, padding='same')(mask)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.2)(fe)
        # normal
        fe = Conv1D(32, 3, strides=2, padding='same')(fe)
        fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.2)(fe)
        # downsample to 7x7
        fe = Conv1D(64, 3, strides=2, padding='same')(fe)
        fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.2)(fe)

        # downsample one more
        fe = Conv1D(128, 3, strides=2, padding='same')(fe)
        fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.2)(fe)

        # flatten feature maps
        fe = Flatten()(fe)
        # real/fake output
        out1 = Dense(1, activation='sigmoid')(fe)

        model = Model(input_text, out1)

        return model


if __name__ == '__main__':
    disc = Discriminator(512, 100)
    disc.get_model()
