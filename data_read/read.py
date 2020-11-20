import json
from pathlib import Path
import logging
import string
from nltk.tokenize import sent_tokenize
import numpy as np
from tqdm import tqdm

from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
tf.compat.v1.disable_eager_execution()


class Dataset:
    def __init__(self, question_path, news_path):
        self.question_path = question_path
        if isinstance(self.question_path, str):
            self.question_path = Path(self.question_path)

        self.news_path = news_path
        if isinstance(self.news_path, str):
            self.news_path = Path(self.news_path)

        self.unique_words = set()
        self.itoc = {}
        self.ctoi = {}
        self.vocab_length = 0
        self.max_len_question = 0
        self.tokenizer = Tokenizer()

        self.relevant_questions_set = []
        self.encoded_relevant_questions_set = []

        self.irrelevant_questions_set = []
        self.encoded_irrelevant_questions_set = []

        self.module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed_fn = self.embed_use()

        self.test_sentences = []

    def get_all_questions(self):
        if not self.question_path.exists():
            raise FileNotFoundError

        with self.question_path.open('r') as data_file:
            data = json.load(data_file)

        if not data:
            raise Exception("data is empty")

        data = data['data']

        for d in data:
            for para in d['paragraphs']:
                for que in para['qas']:
                    self.relevant_questions_set.append(que['question'])

        self.relevant_questions_set = list(set(self.relevant_questions_set))

    def mix_irrelevant_and_crisp_questions(self, news_data, crisp_questions):
        max_data_samples = min(len(crisp_questions), len(news_data))
        logging.info(f"Max data samples possible: {max_data_samples}")

        # remove news item which has empty headline or short_description
        news_data = [n for n in news_data if n['headline'] and n['short_description']]
        news_data = news_data[:max_data_samples]

        for irrelevant, relevant in zip(news_data[:128], crisp_questions[:128]):
            # check for punctuation in headline. If not present, then add full stop
            if irrelevant['headline'][-1] not in string.punctuation:
                irrelevant['headline'] += '.'

            # check for punctuation in question. If not present, then add question mark
            if relevant[-1] not in string.punctuation:
                relevant += '?'

            # sandwich question between news headline and short_description
            dirty_question = irrelevant['headline'] + ' ' + relevant + ' ' + irrelevant['short_description']

            # add some test cases
            if len(self.test_sentences) < 2:
                self.test_sentences.append((dirty_question, relevant))

            self.irrelevant_questions_set.append(dirty_question)

    def prepare_dataset(self):
        news_data = []
        for line in self.news_path.open('r'):
            json_line = json.loads(line)
            news_data.append(json_line)
        logging.info(f"Total data read: {len(news_data)}")

        if not self.relevant_questions_set:
            self.get_all_questions()
        logging.info(f"total crisp questions: {len(self.relevant_questions_set)}")

        self.mix_irrelevant_and_crisp_questions(news_data, self.relevant_questions_set)

        # encode and pad
        self.encode_and_padding()

    def encode_with_use(self, question):
        split_sentences = sent_tokenize(question)

        # get maximum sentences in a question
        if self.max_len_question < len(split_sentences):
            self.max_len_question = len(split_sentences)

        encoded_sentences = self.embed_fn(split_sentences)

        return encoded_sentences

    def encode_and_padding(self):
        # encode irrelevant questions first with USE
        for irrelevant_question in self.irrelevant_questions_set:
            self.encoded_irrelevant_questions_set.append(self.encode_with_use(irrelevant_question))

        # encode relevant question with USE after irrelevant questions are encoded
        for relevant_question in self.relevant_questions_set:
            self.encoded_relevant_questions_set.append(self.encode_with_use(relevant_question))

        # padding for irrelevant question set
        self.encoded_irrelevant_questions_set = pad_sequences(self.encoded_irrelevant_questions_set,
                                                              maxlen=self.max_len_question, dtype='float',
                                                              value=np.zeros(512))

        # padding for relevant question set
        self.encoded_relevant_questions_set = pad_sequences(self.encoded_relevant_questions_set,
                                                            maxlen=self.max_len_question, dtype='float',
                                                            value=np.zeros(512))

        # encode test sample
        encoded_test_sample = []
        for irrelevant_test, relevant_test in self.test_sentences:
            encoded_irrelevant = self.encode_with_use(irrelevant_test)
            encoded_irrelevant = pad_sequences([encoded_irrelevant], maxlen=self.max_len_question, dtype='float',
                                               value=np.zeros(512))
            encoded_relevant = self.encode_with_use(relevant_test)
            encoded_relevant = pad_sequences([encoded_relevant], maxlen=self.max_len_question, dtype='float',
                                             value=np.zeros(512))
            encoded_test_sample.append((encoded_irrelevant, encoded_relevant))
        self.test_sentences = encoded_test_sample

    def embed_use(self):
        with tf.Graph().as_default():
            sentences = tf.compat.v1.placeholder(tf.string)
            embed = hub.Module(self.module_url)
            embeddings = embed(sentences)
            session = tf.compat.v1.train.MonitoredSession()
        return lambda x: session.run(embeddings, {sentences: x})

    def get_vocab(self):
        for que in self.encoded_irrelevant_questions_set:
            all_words = list(set(que.split(' ')))
            self.unique_words.update(all_words)

        # create indices
        self.ctoi = {c: i for i, c in enumerate(self.unique_words)}
        self.itoc = {i: c for i, c in enumerate(self.unique_words)}
        self.vocab_length = len(self.unique_words)
        logging.info(len(self.ctoi))

        return self.unique_words, self.ctoi, self.itoc, self.vocab_length, self.max_len_question


if __name__ == '__main__':
    data = Dataset('../data/squad_v2.json', '../data/News_Category_Dataset_v2.json')
    data.prepare_dataset()
