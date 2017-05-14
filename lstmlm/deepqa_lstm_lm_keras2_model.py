# -*- coding: utf-8 -*-
__author__ = 'kaizh'
import sys

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

from keras import backend as K
from keras.layers import LSTM, SimpleRNN, GRU, Input, Dropout, Dense
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from deepqa_keras2_extentions import CustomEmbedding, DeeqQABaseModel
import numpy as np

import math


def cacc(y_true, y_pred):
    mask = K.max(y_true, axis=-1)
    score = K.equal(K.argmax(y_true, axis=-1),
                    K.argmax(y_pred, axis=-1))
    return K.sum(score * mask) / K.sum(mask)


def categorical_crossentropy(y_true, y_pred):
    mask = K.max(y_true, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -K.sum(mask * y_true * K.log(y_pred), axis=K.ndim(y_pred) - 1)


class RNNLM(DeeqQABaseModel):
    def __init__(self, input_size, voc_size, embedding_size, rnn_outputdim=None, embedding_weight=None,
                 embedding_trainable=True, embedding_writeable=True, rnn_type='lstm', drop_rnn_u=0., drop_rnn_w=0.,
                 drop_embedding=0., drop_rnn=0.):
        self.input_size = input_size
        if rnn_outputdim is None:
            rnn_outputdim = embedding_size

        out_weight_layer_names = []

        inputs = Input(shape=(input_size,), dtype='int32', name='input')

        out_weight_layer_names.append('embedding')
        if embedding_weight is None:
            embedding_weight = [np.random.uniform(low=-0.2, high=0.2, size=(voc_size, embedding_size))]
        embedding_layer = CustomEmbedding(input_dim=voc_size, output_dim=embedding_size, weights=embedding_weight,
                                          mask_zero=True, input_length=input_size, trainable=embedding_trainable,
                                          weights_storeable=embedding_writeable, name=out_weight_layer_names[-1])
        embeddings = embedding_layer(inputs)
        embeddings = Dropout(rate=drop_embedding, name='dropeb')(embeddings)

        out_weight_layer_names.append('rnn')
        if rnn_type == 'lstm':
            rnn_layer = LSTM(units=rnn_outputdim, return_sequences=True, recurrent_dropout=drop_rnn_u,
                             dropout=drop_rnn_w,
                             name=out_weight_layer_names[-1])
        elif rnn_type == 'gru':
            rnn_layer = GRU(units=rnn_outputdim, return_sequences=True, recurrent_dropout=drop_rnn_u,
                            dropout=drop_rnn_w,
                            name=out_weight_layer_names[-1])
        else:
            rnn_layer = SimpleRNN(units=rnn_outputdim, return_sequences=True, recurrent_dropout=drop_rnn_u,
                                  dropout=drop_rnn_w,
                                  name=out_weight_layer_names[-1])
        rnn_outputs = rnn_layer(embeddings)
        rnn_outputs = Dropout(rate=drop_rnn, name='droprnn')(rnn_outputs)

        out_weight_layer_names.append('output')
        dense_layer = TimeDistributed(Dense(units=voc_size, activation='softmax'),
                                      input_shape=(input_size, rnn_outputdim), name=out_weight_layer_names[-1])
        outputs = dense_layer(rnn_outputs)
        model = Model(inputs=inputs, outputs=outputs)

        super(RNNLM, self).__init__(model=model, output_size=voc_size,
                                    output_weight_layer_names=out_weight_layer_names)

    def compile(self, loss_fun=None, opt=None, metrics=None):
        if loss_fun is None:
            loss_fun = categorical_crossentropy
        if metrics is None:
            metrics = [cacc]
        if opt is None:
            opt = Adadelta()
        super(RNNLM, self).compile(loss_fun=loss_fun, opt=opt, metrics=metrics)

    def predict_next(self, sentence_ids):
        """
        :param sentence:
        :return:
        :type sentence list
        """
        return None

    def score(self, input_data, start, end):
        """
        :type input_data:list
        :type start:int
        :type end:int
        """
        inputs = pad_sequences([input_data], self.input_size, padding='post', truncating='post')
        y = self.model.predict(inputs)[0]
        prop = 0
        for i in range(start, min(self.input_size, end)):
            # temp = math.log10(y[i, input_data[i]])
            # print(temp)
            # prop += temp
            print(y[i, input_data[i]])
            prop += math.log10(y[i, input_data[i]])
        return prop

    def batch_score(self, inputs, start_ids, end_ids):
        # type: (list, list, list) -> list
        assert (inputs is not None) and (start_ids is not None) and (end_ids is not None) and len(inputs) > 0 \
               and len(inputs) == len(start_ids) and len(start_ids) == len(end_ids)

        inputs = pad_sequences(inputs, self.input_size, padding='post', truncating='post')
        outpus = self.model.predict(inputs)

        scores = []
        for index, start_id, end_id in zip(range(len(start_ids)), start_ids, end_ids):
            score = 0
            x, y = inputs[index], outpus[index]
            for i in range(start_id, min(end_id, self.input_size)):
                score += math.log10(y[i, x[i]])
            scores.append(score)
        return scores

    def generate(self, start_id):
        """
        :param start_id:
        :return:
        :type start_id:int
        """
        return None


def test_lstm_lm_model_framework():
    """
    :type config:ModelConfig
    """
    batch_size = 2
    input_size = 5
    voc_size = 10000
    embedding_size = 100
    lstm_outputdim = 100

    lm = RNNLM(input_size, voc_size, embedding_size, lstm_outputdim)
    print(lm.model.summary())
    lm.compile()

    inputs = np.random.randint(voc_size, size=(batch_size * 1000, input_size), dtype='int32')
    outputs = np.random.randint(voc_size, size=(batch_size * 1000 * input_size), dtype='int32')
    outputs = np_utils.to_categorical(outputs, voc_size)
    outputs = np.reshape(outputs, (batch_size * 1000, input_size, -1))
    print(outputs.shape)

    lm.model.fit(inputs, outputs, batch_size=batch_size, epochs=100)


if __name__ == '__main__':
    test_lstm_lm_model_framework()
