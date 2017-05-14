# -*- coding: utf-8 -*-
__author__ = 'kaizh'

from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras import backend as K
import codecs
from random import shuffle
import numpy as np
import traceback


def trace_back_log():
    try:
        return traceback.print_exc()
    except:
        return ''


def generator(file, nb_samples, nb_yield_samples, sentence_max_len, to_categorical_size=1,
              start_new=False, padding='pre', reverse=False, fields_count=None):
    """
    :param file:
    :param nb_samples:
    :param nb_yield_samples:
    :param sentence_max_len:
    :param to_categorical_size:
    :param start_new:
    :param padding:
    :return:
    """
    # print file
    if fields_count is None:
        fields_count = -1
    nb_epoch = nb_samples / nb_yield_samples
    # print 'fields_count:{}\tnb_epoch:{}'.format(fields_count, nb_epoch)
    while True:
        batch_data = []
        count_epoch = 0
        # print('start new generator:', file)
        with codecs.open(file, 'r', encoding='utf-8') as f:
            for line in f:
                # print line
                my_instance = parse_1line_4generate_data(line)
                # print my_instance
                if (my_instance is None) or (fields_count > 0 and len(my_instance) != fields_count):
                    continue
                if fields_count < 0:
                    fields_count = len(my_instance)
                # print my_instance
                batch_data.append(my_instance)
                if len(batch_data) >= nb_yield_samples:
                    try:
                        shuffle(batch_data)
                        batch_instance = zip(*batch_data)
                        inputs, outputs = batch_instance[:-1], batch_instance[-1]
                        inputs = [pad_sequences(list(tmp_data), maxlen=sentence_max_len, dtype='int32', padding=padding,
                                                truncating='post', value=0) for tmp_data in inputs]
                        if to_categorical_size == 1:
                            outputs = np.array(list(outputs))
                        else:
                            outputs = np_utils.to_categorical(list(outputs), to_categorical_size)
                    except:
                        inputs, outputs, batch_data = None, None, []
                        print (trace_back_log())

                    if (inputs is None) or (outputs is None):
                        batch_data = []
                        continue
                    if reverse:
                        yield (inputs[::-1], outputs)
                    else:
                        yield (inputs, outputs)
                    batch_data = []
                    count_epoch += 1
                if start_new and count_epoch >= nb_epoch:
                    break
    return


def get_test_data_simple(file_path, max_len):
    extra_info = []
    querys = []
    candidates = []

    with codecs.open(file_path, mode='r', encoding='utf-8') as f:
        for line in f:
            texts = line.rstrip('\r\n').split('\t')
            if texts is None or len(texts) < 3:
                continue

            query_embedding = [int(index) for index in texts[0].split(' ')]
            if query_embedding is None or len(query_embedding) <= 0:
                continue

            cand_embedding = [int(index) for index in texts[1].split(' ')]
            if cand_embedding is None or len(cand_embedding) <= 0:
                continue

            querys.append(query_embedding)
            candidates.append(cand_embedding)
            extra_info.append('%s\t%s\t%s' % (texts[0], texts[1], texts[2]))

    querys = pad_sequences(querys, maxlen=max_len, dtype='int32', padding='pre', truncating='post', value=0)
    candidates = pad_sequences(candidates, maxlen=max_len, dtype='int32', padding='pre', truncating='post', value=0)
    return extra_info, [querys, candidates]


def get_test_data(file_path, word_dict, max_len, unk=None, padding='pre', reverse=False, contain_label=True):
    extra_info = []
    querys = []

    if (unk is not None) and (unk in word_dict):
        unk = word_dict[unk]
        print(unk)
    else:
        unk = None

    with codecs.open(file_path, mode='r', encoding='utf-8') as f:
        for line in f:
            texts = line.rstrip('\r\n').split('\t')
            if texts is None:
                continue

            length = len(texts)
            if (contain_label):
                length -= 1
            if length < 1:
                continue

            embeddings = [get_embedding(texts[index].split(' '), word_dict, unk) for index in range(length)]
            can_add = True
            for embedding in embeddings:
                if embedding is None or len(embedding) <= 0:
                    can_add = False
                    break
            if can_add:
                querys.append(embeddings)
                extra_info.append(line)

    querys = [pad_sequences(query, maxlen=max_len, dtype='int32', padding=padding, truncating='post', value=0) for query
              in zip(*querys)]
    if reverse:
        return extra_info, querys[::-1]
    else:
        return extra_info, querys


def get_embedding(words, dict, unk):
    query_embedding = []
    for word in words:
        if word in dict:
            query_embedding.append(dict[word])
        elif unk is not None:
            query_embedding.append(unk)
    return query_embedding


def load_word_dict(file_path, add_padding_vec=False):
    word_dict = {}
    if add_padding_vec:
        word_dict['<pad>'] = 0
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if line == '':
                continue
            fields = line.split('\t')
            if len(fields) != 2:
                print('error line:', line)
            word, word_id = fields
            if add_padding_vec:
                word_dict[word] = int(word_id) + 1
            else:
                word_dict[word] = int(word_id)
            print word_id
    print('word count of word dict : %d' % len(word_dict))
    return word_dict


def load_w2v_model(file_path, add_padding_vec=False):
    word_dict = {}
    embeddings = []
    voc_size, vec_size = 0, 0
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            line = line.rstrip('\r\n')
            if line == '':
                continue
            fields = line.split(' ')
            if len(fields) <= 1:
                continue
            if count == 0:
                voc_size = int(fields[0])
                vec_size = int(fields[1])
                if add_padding_vec:
                    voc_size += 1
                    word_dict['<pad>'] = len(embeddings)
                    embeddings.append([0. for i in range(vec_size)])
            else:
                word, vec = fields[0], [float(float_str) for float_str in fields[1:vec_size + 1]]
                word_dict[word] = len(embeddings)
                embeddings.append(vec)
            count += 1
    assert (len(word_dict) == voc_size and len(embeddings) == voc_size)
    print('word count of word dict : %d' % len(word_dict))
    return vec_size, word_dict, [np.asarray(embeddings)]


def parse_1line_4generate_data(line):
    try:
        texts = line.rstrip('\r\n').split('\t')
        if len(texts) < 3:
            return None
        if len(texts[-1]) != 1:
            return None
        target = int(texts[-1])
        if target != 1 and target != 0:
            return None
        data = []
        for field in texts[:-1]:
            tmp = [int(word_id) for word_id in field.split(' ') if word_id != '']
            if len(tmp) <= 0:
                return None
            data.append(tmp)
        data.append(target)
        return data
    except:
        print(trace_back_log())
    return None


def get_dense_init_weights(activation, n_in, n_out):
    low = -np.sqrt(6. / (n_in + n_out))
    high = np.sqrt(6. / (n_in + n_out))
    w_weight = np.random.uniform(low=low, high=high, size=(n_in, n_out))
    if activation != 'tanh':
        w_weight *= 4
    bias_weight = np.zeros((n_out,), dtype=K.floatx())
    return [w_weight, bias_weight]


class ModelConfig(object):
    def __init__(self):
        self.training_data_fields_count = None

        self.model_save_path = None
        self.training_data_file = None
        self.validation_data_file = None

        self.predict_data_file = None
        self.predict_out_file = None

        self.word_dict = None

        self.nb_epoch = 30
        self.batch_size = 256
        self.train_batch_count = 128
        self.validation_bath_count = 128
        self.samples_per_epoch = self.batch_size * self.train_batch_count
        self.nb_val_samples = self.batch_size * self.validation_bath_count

        self.input_size = 100

        self.embedding_size = 100
        self.embedding_weight = None
        self.embedding_init_file = None
        self.embedding_trainable = True
        self.embedding_writeable = True
        self.drop_embedding = 0.3

        self.gru_embedding_size = 100
        self.drop_gru_u = 0.
        self.drop_gru_w = 0.
        self.drop_gru = 0.3

        self.attention_inner_embedding_size = 10

        self.drop_cnn = 0.3
        self.nb_filter = 100
        self.filter_length = 2

        self.tensor_count = 5

        self.output_size = 1
        self.out_active = 'sigmoid'

        self.input_reverse = False
        self.bidirectional = False

        self.meta = {}

    def show_summary(self):
        config_dict = {'model save path': self.model_save_path, 'training data file': self.training_data_file,
                       'validataion data file': self.validation_data_file, 'reverse input': self.input_reverse,
                       'predict data file': self.predict_data_file, 'predict out file': self.predict_out_file,
                       'epoch number': self.nb_epoch, 'batch size': self.batch_size,
                       'train batch count': self.train_batch_count, 'tensor count': self.tensor_count,
                       'validation bath count': self.validation_bath_count, 'input size': self.input_size,
                       'embedding size': self.embedding_size, 'embedding init file': self.embedding_init_file,
                       'gru embedding size': self.gru_embedding_size,
                       'attention inner embedding size': self.attention_inner_embedding_size,
                       'embedding trainable': self.embedding_trainable, 'bidirectional': self.bidirectional,
                       'embedding writeable': self.embedding_writeable,
                       'training data fields count': self.training_data_fields_count}

        for key, value in self.meta.items():
            config_dict[key] = value

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
                              ['debug', 'train', 'test', 'predict', 'save', 'bi', 'eb_size=', 'in_size=', 'gru_size='])
    function_name = ''
    for op, value in opt:
        if op == '-m':
            config.model_save_path = value
            config.predict_out_file = value + '.predict.txt'
        elif op == '-t':
            config.training_data_file = value
        elif op == '-v':
            config.validation_data_file = value
        elif op == '-p':
            config.predict_data_file = value
        elif op == '-o':
            config.predict_out_file = value
        elif op == '-e':
            config.nb_epoch = int(value)
        elif op == '-b':
            config.batch_size = int(value)
        elif op == '-d':
            config.word_dict = load_word_dict(value)
        elif op == '-c':
            config.tensor_count = int(value)
        elif op == '-i':
            config.embedding_init_file = value
        elif op == '-s':
            config.embedding_trainable = False
        elif op == '-w':
            config.embedding_writeable = False
        elif op == '-r':
            config.input_reverse = True
        elif op.startswith('--'):
            if op == '--bi':
                config.bidirectional = True
            if op == "--eb_size":
                config.embedding_size = int(value)
            if op == "--in_size":
                config.input_size = int(value)
            if op == '--gru_size':
                config.gru_embedding_size = int(value)
            else:
                function_name = op.lstrip('-')
        else:
            config.meta[op.lstrip('-')] = value

    config.samples_per_epoch = config.batch_size * config.train_batch_count
    config.nb_val_samples = config.batch_size * config.validation_bath_count
    config.show_summary()
    function_name = function_name.lower()
    return function_name, config
