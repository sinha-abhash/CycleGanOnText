import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, LayerNormalization, Masking
from tensorflow.keras.models import Model

import numpy as np

import logging

logging.basicConfig(level=logging.INFO)


class Generator:
    def __init__(self, embed_size, max_len_question):
        self.embed_size = embed_size
        self.max_len_question = max_len_question
        self.output_dim = 1

    @staticmethod
    def universal_embedding(x):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        embed = hub.Module(module_url)
        return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

    def get_model(self):
        input_text = Input(shape=(self.max_len_question, self.embed_size))
        mask = Masking(mask_value=np.zeros(self.embed_size))(input_text)

        # lstm1
        lstm1 = LSTM(256, return_sequences=True, return_state=True)
        lstm1_output, state_h1, state_c1 = lstm1(mask)
        layer_norm1 = LayerNormalization(axis=1)(lstm1_output)

        # lstm2
        lstm2 = LSTM(128, return_sequences=True, return_state=True)
        lstm2_output, state_h2, state_c2 = lstm2(layer_norm1)
        layer_norm2 = LayerNormalization(axis=1)(lstm2_output)

        # lstm3
        lstm3 = LSTM(64, return_sequences=True, return_state=True)
        lstm3_output, state_h3, state_c3 = lstm3(layer_norm2)
        layer_norm3 = LayerNormalization(axis=1)(lstm3_output)

        # lstm4
        lstm4 = LSTM(128, return_sequences=True, return_state=True)
        lstm4_output, state_h4, state_c4 = lstm4(layer_norm3)
        layer_norm4 = LayerNormalization(axis=1)(lstm4_output)

        # lstm5
        lstm5 = LSTM(256, return_sequences=True, return_state=True)
        lstm5_output, state_h5, state_c5 = lstm5(layer_norm4)
        layer_norm5 = LayerNormalization(axis=1)(lstm5_output)

        # lstm6
        lstm6 = LSTM(self.embed_size, return_sequences=True, return_state=True, activation='tanh')
        lstm6_output, state_h6, state_c6 = lstm6(layer_norm5)

        model = Model(inputs=input_text, outputs=lstm6_output)

        return model


if __name__ == '__main__':
    gen = Generator(512, 100)
    gen.get_model()
