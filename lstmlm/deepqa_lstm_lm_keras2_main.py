# -*- coding: utf-8 -*-
__author__ = 'kaizh'
import sys

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

from deepqa_lstm_lm_keras2_utils import init_lstm_lm_model, ModelConfig, parse_config, rnn_generator

import codecs


def test_gen(config):
    """
    :type config:ModelConfig
    """
    import locale

    model = init_lstm_lm_model(config, False)
    model.compile()
    model.model.load_weights(config.model_path)
    model.print_embedding()

    word_dict = config.word_dict
    assert '<unk>' in word_dict and '<tag>' in word_dict and '<eos>' in word_dict
    unk_id, tag_id, eos_id = word_dict['<unk>'], word_dict['<tag>'], word_dict['<eos>']
    while True:
        input = raw_input('input :').strip().decode(
            sys.stdin.encoding or locale.getpreferredencoding(True))
        if len(input) == 0:
            break
        fileds = input.split(' ')
        if len(fileds) != 2:
            continue
        q, a = fileds
        q = [word_dict.get(char, unk_id) for char in q]
        a = [word_dict.get(char, unk_id) for char in a]
        print q
        print a
        score = model.score(q + [tag_id] + a + [eos_id], len(q) + 1, len(q) + len(a) + 2)
        print('score : {}'.format(score))


def train_lstm_lm_model(config):
    """
    :type config:ModelConfig
    """

    model = init_lstm_lm_model(config, True)
    model.compile()

    model.print_embedding()
    voc_size = len(config.word_dict)

    model.fit_generator(model_save_path=config.model_path,
                        epochs=config.nb_epoch, steps_per_epoch=config.train_batch_count,
                        generator_train=rnn_generator(file_path=config.training_data_file,
                                                      category_size=voc_size, word_dict=config.word_dict,
                                                      sen_max_len=config.input_size, batch_size=config.batch_size,
                                                      steps_per_epoch=config.train_batch_count,
                                                      batch_step_count=config.batch_step_count,
                                                      shuffle=True, eos_id=config.eos_id,
                                                      start_new_per_epoch=True),
                        validation_steps=config.validation_bath_count,
                        validation_data=rnn_generator(file_path=config.validation_data_file,
                                                      category_size=voc_size, word_dict=config.word_dict,
                                                      sen_max_len=config.input_size, batch_size=config.batch_size,
                                                      steps_per_epoch=config.validation_bath_count,
                                                      batch_step_count=config.batch_step_count,
                                                      shuffle=False, eos_id=config.eos_id,
                                                      start_new_per_epoch=True))
    model.print_embedding()


def test_lstm_lm_model(config):
    """
    :type config:ModelConfig
    """
    import string
    def load_test_data(filename):
        """
        :type filename:str
        """
        datas = []
        fin = codecs.open(filename, 'r', 'utf-8')
        strip_chars = string.whitespace + '\r\n'
        for line in fin:
            line = line.strip(strip_chars)
            if len(line) == 0:
                continue
            fields = line.split('\t')
            if len(fields) != 2:
                continue
            datas.append(fields)
        fin.close()
        return datas

    def convert_word2id(datas, word_dict):
        """
        :type datas:list
        :type word_dict:dict
        """
        assert '<unk>' in word_dict and '<tag>' in word_dict and '<eos>' in word_dict
        unk_id, tag_id, eos_id = word_dict['<unk>'], word_dict['<tag>'], word_dict['<eos>']
        inputs, start_ids, end_ids = [], [], []
        for data in datas:
            q, a = data
            q = [word_dict.get(char, unk_id) for char in q]
            a = [word_dict.get(char, unk_id) for char in a]
            assert len(q) > 0 and len(a) > 0
            inputs.append(q + [tag_id] + a + [eos_id])
            start_ids.append(len(q) + 1)
            end_ids.append(len(inputs))
        return inputs, start_ids, end_ids

    model = init_lstm_lm_model(config, False)
    model.compile()
    model.model.load_weights(config.model_path)
    model.print_embedding()

    datas = load_test_data(config.test_data_file)
    inputs, start_ids, end_ids = convert_word2id(datas, config.word_dict)
    scores = model.batch_score(inputs, start_ids, end_ids)

    fout = codecs.open(config.output_file, 'w', 'utf-8')
    for data, score in zip(datas, scores):
        fout.write(data[0] + '\t' + data[1] + '\t' + str(score) + '\r\n')
    fout.close()


def save_lstm_lm_model(config):
    """
    :type config:ModelConfig
    """
    model = init_lstm_lm_model(config, False)
    model.compile()
    model.model.load_weights(config.model_path)
    if config.embedding_writeable:
        index2word = [None for i in range(len(config.word_dict))]
        for key, value in config.word_dict.items():
            index2word[value] = key
        model.save_embedding(config.model_path + '.embed.txt', config.embedding_size, index2word)
    model.save_weights(config.model_path + '.weight.txt')


if __name__ == '__main__':

    config = ModelConfig()

    function_name, config = parse_config(config)

    if function_name == 'train':
        #assert (config.model_path is not None)
        #assert (config.training_data_file is not None)
        #assert (config.validation_data_file is not None)
        train_lstm_lm_model(config)
    elif function_name == 'test':
        test_lstm_lm_model(config)
    elif function_name == 'save':
        save_lstm_lm_model(config)
    elif function_name == 'gen':
        test_gen(config)
