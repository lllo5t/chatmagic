# -*- coding: utf-8 -*-
__author__ = 'kaizh'
import sys

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from deepqa_keras2_utils import load_word_dict, load_w2v_model
import numpy as np
from deepqa_lstm_lm_keras2_model import RNNLM

import codecs
import random


def rnn_generator(file_path, category_size, word_dict, sen_max_len, batch_size, steps_per_epoch, batch_step_count,
                  shuffle, start_new_per_epoch, eos_id=None):
    """
    :type start_new_per_epoch:bool
    """

    def process_batch(batch_data):
        inputs, outputs = [item[:-1] for item in batch_data], [item[1:] for item in batch_data]
        inputs = pad_sequences(inputs, maxlen=sen_max_len, padding='post', truncating='post')
        outputs = pad_sequences(outputs, maxlen=sen_max_len, padding='post')
        outputs = [np_utils.to_categorical(outputs[i], category_size) for i in range(outputs.shape[0])]
        for s in range(len(outputs)):
            for t in range(sen_max_len):
                if outputs[s][t, 0] == 1:
                    outputs[s][t, 0] = 0
        outputs = np.array(outputs)
        return inputs, outputs

    datas = []
    global_step_count, local_step_count = 0, 0
    unk_id = word_dict['<unk>']
    while True:
        fin = codecs.open(file_path, 'r', encoding='utf-8')
        for line in fin:
            line = line.rstrip('\r\n')
            if len(line) == 0:
                continue
            # print line
            line = [word_dict.get(item, unk_id) for item in line.split(' ')]
            if eos_id is not None:
                line.append(eos_id)
            datas.append(line)
            if len(datas) % batch_size == 0:
                global_step_count += 1
                local_step_count += 1
                if local_step_count == batch_step_count or global_step_count == steps_per_epoch:
                    assert len(datas) == local_step_count * batch_size
                    if shuffle:
                        datas = random.sample(datas, len(datas))
                    for i in range(local_step_count):
                        data = datas[i * batch_size: (i + 1) * batch_size]
                        inputs, outputs = process_batch(data)
                        yield inputs, outputs
                    datas = []
                    local_step_count = 0
                    if global_step_count == steps_per_epoch:
                        global_step_count = 0
                        if start_new_per_epoch:
                            break
        fin.close()



def init_lstm_lm_model(config, is_train=True):
    """
    :type is_train: bool
    :type config ModelConfig
    :rtype RNNLM
    """

    embedding_weight = None
    if config.word_dict is not None and config.embedding_weight is not None:
        embedding_weight = config.embedding_weight
    elif config.embedding_init_file is not None:
        from datetime import datetime
        print(str(datetime.now()) + " start loading Word2Vec")
        config.embedding_size, config.word_dict, embedding_weight = load_w2v_model(config.embedding_init_file,
                                                                                   config.add_extra_pad)
        print(str(datetime.now()) + " end loading Word2Vec")
        if config.eos_key is not None:
            config.eos_id = config.word_dict[config.eos_key]
        else:
            config.eos_id = 'unknown'
        print('dict length : {}, vec size : {}, unk id : {}, eos id : {}'.format(len(config.word_dict),
                                                                                 config.embedding_size,
                                                                                 config.word_dict['<unk>'],
                                                                                 config.eos_id))

    if config.eos_key is not None:
        config.eos_id = config.word_dict[config.eos_key]
    else:
        config.eos_id = None

    voc_size = len(config.word_dict)

    drop_embedding = 0
    drop_rnn_u = 0
    drop_rnn_w = 0
    drop_rnn = 0
    if is_train:
        drop_embedding = config.drop_embedding
        drop_rnn_u = config.drop_rnn_u
        drop_rnn_w = config.drop_rnn_w
        drop_rnn = config.drop_rnn

    model = RNNLM(input_size=config.input_size, voc_size=voc_size, embedding_size=config.embedding_size,
                  rnn_type=config.rnn_type, rnn_outputdim=config.rnn_embedding_size, embedding_weight=embedding_weight,
                  embedding_trainable=config.embedding_trainable, embedding_writeable=config.embedding_writeable,
                  drop_embedding=drop_embedding, drop_rnn_u=drop_rnn_u, drop_rnn_w=drop_rnn_w, drop_rnn=drop_rnn)

    print(model.model.summary())
    return model


class ModelConfig(object):
    def __init__(self):
        self.eos_key = None
        self.eos_id = None
        self.model_path = None
        self.training_data_file = None
        self.validation_data_file = None
        self.test_data_file = None
        self.output_file = None

        self.word_dict = None
        self.add_extra_pad = False

        self.nb_epoch = 50
        self.batch_size = 128
        self.batch_step_count = 1000
        self.train_batch_count = 50000
        self.validation_bath_count = 10000

        self.input_size = 50

        self.embedding_size = 512
        self.embedding_weight = None
        self.embedding_init_file = None
        self.embedding_trainable = True
        self.embedding_writeable = True
        self.drop_embedding = 0.5

        self.rnn_type = 'lstm'
        self.rnn_embedding_size = self.embedding_size
        self.drop_rnn_u = 0.
        self.drop_rnn_w = 0.
        self.drop_rnn = 0.5

        self.input_reverse = False

    def show_summary(self):
        config_dict = {'model save path': self.model_path, 'training data file': self.training_data_file,
                       'validataion data file': self.validation_data_file, 'reverse input': self.input_reverse,
                       'test data file': self.test_data_file, 'output file': self.output_file,
                       'epoch number': self.nb_epoch, 'batch size': self.batch_size,
                       'batch cache': self.batch_step_count, 'eos key': self.eos_key,
                       'train batch count': self.train_batch_count, 'validation bath count': self.validation_bath_count,
                       'input size': self.input_size, 'rnn type': self.rnn_type,
                       'embedding size': self.embedding_size, 'embedding init file': self.embedding_init_file,
                       'rnn embedding size': self.rnn_embedding_size, 'add extra pad id': self.add_extra_pad,
                       'embedding trainable': self.embedding_trainable,
                       'embedding writeable': self.embedding_writeable}

        for key in config_dict.keys():
            if config_dict[key] is not None:
                print('\t{} : {}'.format(key, config_dict[key]))

        return config_dict


def parse_config(config=None):
    """
    :type config:ModelConfig
    """
    import sys
    import getopt
    # import theano
    #
    # print(theano.config)

    if config is None:
        config = ModelConfig()
    opt, args = getopt.getopt(sys.argv[1:], 'm:t:v:p:o:e:b:d:c:i:swr',
                              ['debug', 'gen', 'train', 'test', 'predict', 'save', 'epoch=', 'eb_size=', 'in_size=',
                               'pad', 'eos=', 'rnn_size=', 'train_step=', 'valid_step=', 'batch_cache='])
    function_name = ''
    for op, value in opt:
        if op == '-m':
            config.model_path = value
            config.output_file = config.model_path + '.result'
        elif op == '-t':
            config.training_data_file = value
        elif op == '-v':
            config.validation_data_file = value
        elif op == '-p':
            config.test_data_file = value
        elif op == '-o':
            config.output_file = value
        elif op == '-b':
            config.batch_size = int(value)
        elif op == '-d':
            config.word_dict = value
        elif op == '-i':
            config.embedding_init_file = value
        elif op == '-s':
            config.embedding_trainable = False
        elif op == '-w':
            config.embedding_writeable = False
        elif op == '-r':
            config.input_reverse = True
        elif op.startswith('--'):
            if op == "--epoch":
                config.nb_epoch = int(value)
            elif op == '--pad':
                config.add_extra_pad = True
            elif op == '--eos':
                config.eos_key = value
            elif op == '--batch_cache':
                config.batch_step_count = int(value)
            elif op == '--train_step':
                config.train_batch_count = int(value)
            elif op == '--valid_step':
                config.validation_bath_count = int(value)
            elif op == "--eb_size":
                config.embedding_size = int(value)
            elif op == "--in_size":
                config.input_size = int(value)
            elif op == '--rnn_size':
                config.rnn_embedding_size = int(value)
            else:
                function_name = op.lstrip('-')

    if config.word_dict is not None:
        config.word_dict = load_word_dict(config.word_dict, config.add_extra_pad)

    config.show_summary()
    function_name = function_name.lower()
    return function_name, config

