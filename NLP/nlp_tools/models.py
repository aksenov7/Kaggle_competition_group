from abc import abstractmethod
import typing as t
import logging

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from consecution import Node
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense

from . import bert_tokenizer


class ModelNode(Node):
    def __init__(
        self,
        name: str,
        train_target_key,
        train_data_key,
        predict_data_key,
        model_settings: t.Dict[str, t.Any] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)

        if model_settings is None:
            model_settings = {}

        self._model = self.model(**model_settings)
        self.predict_data_key = predict_data_key
        self.train_data_key = train_data_key
        self.train_target_key = train_target_key

    @property
    @abstractmethod
    def model(self):
        pass

    def process(self, item):
        logging.info(f'Узел {self.name} вида {self.__class__.__name__} начал обучение')

        x = self.global_state.processing_data[self.train_data_key]
        trained_model = self._model.fit(x, self.global_state.processing_data[self.train_target_key])

        predicted = trained_model.predict(self.global_state.processing_data[self.predict_data_key])

        result = pd.DataFrame(data=dict(
            target=predicted
        ))
        result.to_csv(f'results/{self.name}_result.csv', index=False)

        logging.info(f'Узел {self.name} вида {self.__class__.__name__} закончил работу')

        self._push(item)


class MultinomialNBNode(ModelNode):
    @property
    def model(self):
        return MultinomialNB


class TensorFlowCustomBinaryNode(Node):
    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 250
    EMBEDDING_DIM = 100

    def __init__(
        self,
        name: str,
        train_target_key,
        train_data_key,
        predict_data_key,
        model_layers,
        **kwargs
    ):
        super().__init__(name, **kwargs)

        self._tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

        self.predict_data_key = predict_data_key
        self.train_data_key = train_data_key
        self.train_target_key = train_target_key
        self.model_layers = model_layers

    def process(self, item):
        logging.info(f'Узел {self.name} вида {self.__class__.__name__} начал обучение')

        x = self.global_state.processing_data[self.train_data_key].values

        self._tokenizer.fit_on_texts(x)
        X = self._tokenizer.texts_to_sequences(x)
        X = pad_sequences(X, maxlen=self.MAX_SEQUENCE_LENGTH)
        Y = self.global_state.processing_data[self.train_target_key]

        epochs = 1
        batch_size = 32

        self._model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.MAX_NB_WORDS, self.EMBEDDING_DIM, input_length=X.shape[1]),
            *self.model_layers,
            tf.keras.layers.Dense(1)
        ])

        self._model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            optimizer=tf.keras.optimizers.Adam(1e-4),
                            metrics=['accuracy'])

        logging.info(self._model.summary())

        history = self._model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

        seq = self._tokenizer.texts_to_sequences(self.global_state.processing_data[self.predict_data_key])
        padded = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
        predicted = self._model.predict(padded)

        result = [1 if x >= 0.5 else 0 for x in predicted]
        result = pd.DataFrame(result)
        result.to_csv(f'results/{self.name}_result.csv', index=False)

        logging.info(f'Узел {self.name} вида {self.__class__.__name__} закончил работу')

        self._push(item)


class TensorFlowLSTMNBinaryNode(Node):
    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 250
    EMBEDDING_DIM = 100

    def __init__(
        self,
        name: str,
        train_target_key,
        train_data_key,
        predict_data_key,
        **kwargs
    ):
        super().__init__(name, **kwargs)

        self._tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

        self.predict_data_key = predict_data_key
        self.train_data_key = train_data_key
        self.train_target_key = train_target_key

    def process(self, item):
        logging.info(f'Узел {self.name} вида {self.__class__.__name__} начал обучение')

        x = self.global_state.processing_data[self.train_data_key].values

        self._tokenizer.fit_on_texts(x)
        X = self._tokenizer.texts_to_sequences(x)
        X = pad_sequences(X, maxlen=self.MAX_SEQUENCE_LENGTH)
        Y = self.global_state.processing_data[self.train_target_key]

        epochs = 1
        batch_size = 64

        self._model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.MAX_NB_WORDS, self.EMBEDDING_DIM, input_length=X.shape[1]),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self._model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            optimizer=tf.keras.optimizers.Adam(1e-4),
                            metrics=['accuracy'])

        logging.info(self._model.summary())

        history = self._model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                                  callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        seq = self._tokenizer.texts_to_sequences(self.global_state.processing_data[self.predict_data_key])
        padded = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
        predicted = self._model.predict(padded)

        result = [1 if x >= 0.5 else 0 for x in predicted]
        result = pd.DataFrame(result)
        result.to_csv(f'results/{self.name}_result.csv', index=False)

        logging.info(f'Узел {self.name} вида {self.__class__.__name__} закончил работу')

        self._push(item)


class TensorFlowCNNBinaryNode(Node):
    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 250
    EMBEDDING_DIM = 100

    def __init__(
        self,
        name: str,
        train_target_key,
        train_data_key,
        predict_data_key,
        **kwargs
    ):
        super().__init__(name, **kwargs)

        self._tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

        self.predict_data_key = predict_data_key
        self.train_data_key = train_data_key
        self.train_target_key = train_target_key

    def process(self, item):
        logging.info(f'Узел {self.name} вида {self.__class__.__name__} начал обучение')

        x = self.global_state.processing_data[self.train_data_key].values

        self._tokenizer.fit_on_texts(x)
        X = self._tokenizer.texts_to_sequences(x)
        X = pad_sequences(X, maxlen=self.MAX_SEQUENCE_LENGTH)
        Y = self.global_state.processing_data[self.train_target_key]

        epochs = 1
        batch_size = 32

        self._model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.MAX_NB_WORDS, self.EMBEDDING_DIM, input_length=X.shape[1]),
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self._model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            optimizer=tf.keras.optimizers.Adam(1e-4),
                            metrics=['accuracy'])
        logging.info(self._model.summary())
        history = self._model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

        seq = self._tokenizer.texts_to_sequences(self.global_state.processing_data[self.predict_data_key])
        padded = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
        predicted = self._model.predict(padded)

        result = [1 if x >= 0.5 else 0 for x in predicted]
        result = pd.DataFrame(result)
        result.to_csv(f'results/{self.name}_result.csv', index=False)

        logging.info(f'Узел {self.name} вида {self.__class__.__name__} закончил работу')

        self._push(item)


class BertBinaryNode(Node):

    @staticmethod
    def bert_encode(texts, tokenizer, max_len=512):
        all_tokens = []
        all_masks = []
        all_segments = []

        for text in texts:
            text = tokenizer.tokenize(text)

            text = text[:max_len - 2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = max_len - len(input_sequence)

            tokens = tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_len

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    @staticmethod
    def build_model(bert_layer, max_len=512):
        input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

        _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        out = Dense(1, activation='sigmoid')(clf_output)

        model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        model.compile(tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])

        return model

    MODULE_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

    def __init__(
        self,
        name: str,
        train_target_key,
        train_data_key,
        predict_data_key,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.predict_data_key = predict_data_key
        self.train_data_key = train_data_key
        self.train_target_key = train_target_key

    def process(self, item):
        logging.info(f'Узел {self.name} вида {self.__class__.__name__} начал обучение')

        x = self.global_state.processing_data[self.train_data_key]
        y = self.global_state.processing_data[self.train_target_key]

        bert_layer = hub.KerasLayer(self.MODULE_URL, trainable=True)

        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = bert_tokenizer.FullTokenizer(vocab_file, do_lower_case)

        train_input = self.bert_encode(x.values, tokenizer, max_len=160)
        predict_input = self.bert_encode(
            self.global_state.processing_data[self.predict_data_key].values, tokenizer, max_len=160
        )
        train_labels = y.values

        model = self.build_model(bert_layer, max_len=160)
        logging.info(model.summary())

        train_history = model.fit(
            train_input, train_labels,
            validation_split=0.2,
            epochs=5,
            batch_size=32
        )

        test_pred = model.predict(predict_input)

        result = pd.DataFrame(test_pred.round().astype(int))
        result.to_csv(f'results/{self.name}_result.csv', index=False)

        logging.info(f'Узел {self.name} вида {self.__class__.__name__} закончил работу')

        self._push(item)
