# -*- coding: utf-8 -*-
__author__ = 'kaizh'

from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, LearningRateScheduler
from keras.layers import Embedding, Dense, Layer, MaxPooling1D, InputSpec, GRU, Recurrent # , AveragePooling1D
from keras.layers.recurrent import _time_distributed_dense
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers, constraints, activations, initializers
import codecs
import numpy as np
from datetime import datetime

class SaveCallback(Callback):
    def __init__(self, model_file, simple_info=True):
        super(Callback, self).__init__()
        self._model_file = model_file
        self._simple_info = simple_info

    def on_epoch_end(self, epoch, logs={}):
        cur_model_file = self._model_file
        if not self._simple_info:
            cur_model_file = '{}.{}.acc-{:.4f}.loss-{:.4f}'.format(self._model_file, epoch, logs['acc'], logs['loss'])
            val_acc = logs.get('val_acc')
            if val_acc is not None:
                cur_model_file = '{}.val_acc-{:.4f}'.format(cur_model_file, val_acc)
            val_loss = logs.get('val_loss')
            if val_loss is not None:
                cur_model_file = '{}.val_loss-{:.4f}'.format(cur_model_file, val_loss)
            cur_model_file = '{}.model'.format(cur_model_file)
        self.model.save_weights(cur_model_file, overwrite=True)


class SaveEarlyStoppingCallback(EarlyStopping):
    def __init__(self, model_file, monitor='val_loss', patience=0, verbose=0):
        super(SaveEarlyStoppingCallback, self).__init__(
            monitor=monitor, patience=patience, verbose=verbose)
        self.model_file = model_file

    def on_epoch_end( self, epoch, logs=None ):
        if logs is None:
            logs = {}
        super(SaveEarlyStoppingCallback, self).on_epoch_end(epoch, logs=logs)
        current = logs.get(self.monitor)
        if current is not None and current == self.best:
            self.model.save_weights(self.model_file, overwrite=True)


class DeeqQABaseModel(object):
    def __init__(self, model, output_size, output_weight_layer_names):
        """
        :type model:Model
        """
        self.output_size = output_size
        self.out_weight_layer_names = output_weight_layer_names
        self.model = model
        super(DeeqQABaseModel, self).__init__()

    def complie(self, loss_fun=None, opt=None, metrics=None):
        """
        complie is a typo. save this func for Compatibility
        """
        self.compile(loss_fun, opt, metrics)

    def compile(self, loss_fun=None, opt=None, metrics=None):
        # type: (object, object, object) -> None
        if metrics is None:
            metrics = ['accuracy']
        if opt is None:
            opt = Adam()
        if loss_fun is None:
            loss_fun = 'binary_crossentropy'
            if self.output_size > 1:
                loss_fun = 'categorical_crossentropy'
        print('start compile model:{}'.format(datetime.now()))
        self.model.compile(loss=loss_fun, optimizer=opt, metrics=metrics)
        print('end  compile  model:{}'.format(datetime.now()))

    def train( self, model_save_path, generator_train, steps_per_epoch, epochs, validation_data, validation_steps ):
        self.fit_generator(model_save_path, generator_train, steps_per_epoch, epochs, validation_data, validation_steps)

    def fit_generator( self, model_save_path, generator_train, steps_per_epoch, epochs, validation_data, validation_steps ):
        self.model.fit_generator(generator=generator_train, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                 validation_data=validation_data, validation_steps=validation_steps,
                                 callbacks=[SaveCallback(model_save_path + '.last'),
                                            SaveEarlyStoppingCallback(model_save_path + '.best', patience=5)])

    def fit( self, model_save_path, data, batch_size=32, epochs=10, validation_split=0., validation_data=None,
             shuffle=True, class_weight=None, sample_weight=None ):
        inputs, outputs = data
        self.model.fit(x=inputs, y=outputs, batch_size=batch_size, epochs=epochs,
                       validation_split=validation_split, validation_data=validation_data,
                       shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight,
                       callbacks=[SaveCallback(model_save_path + '.last'),
                                  SaveEarlyStoppingCallback(model_save_path + '.best', patience=5)])

    def predict(self, x, y=None, extra_info=None, group_ids=None, output='result.txt'):
        assert (self.model is not None)
        print('start predict:', str(datetime.now()))
        f = codecs.open(output, mode='w', encoding='UTF-8')
        predicts = self.model.predict(x)
        if group_ids is not None:
            f.write('GroupId\t')
        if extra_info is not None:
            f.write('query\tcandidate\t')
        if y is not None:
            f.write('Lable\t')
        f.write('Predict\n')
        for i in range(len(predicts)):
            if group_ids is not None:
                f.write(str(group_ids[i]) + '\t')
            if extra_info is not None:
                f.write(extra_info[i])
                f.write('\t')
            if y is not None:
                f.write(str(y[i]) + '\t')
            p = predicts[i][0]
            if self.output_size > 1:
                p = predicts[i][1]
            f.write(str(p) + '\n')
        f.close()
        print('end predict:', str(datetime.now()))

    def save_weights(self, model_path):
        f = open(model_path, "w")
        for layer_name in self.out_weight_layer_names:
            print(layer_name)
            weights = self.model.get_layer(name=layer_name).get_weights()
            for weight in weights:
                print('\t%s\t%s' % (str(len(weight.shape)), str(weight.shape)))
                if len(weight.shape) == 1:
                    np.savetxt(f, weight.reshape(1, weight.shape[0]), delimiter=' ')
                elif len(weight.shape) == 2:
                    np.savetxt(f, weight, delimiter=' ')

    def save_embedding(self, embedding_path, embedding_size, index2word, name='embedding'):
        embeddings = self.model.get_layer(name=name).get_weights()
        f = open(embedding_path, 'w')
        for weight in embeddings:
            print(len(weight.shape), weight.shape)
            if len(weight.shape) == 1:
                np.savetxt(f, weight.reshape(1, weight.shape[0]), delimiter=' ', fmt='%f')
            else:
                np.savetxt(f, weight, delimiter=' ', fmt='%f')
        vector_strs = []
        with open(embedding_path, 'r') as f:
            for line in f:
                if len(line) > 0:
                    vector_strs.append(line)

        f = codecs.open(embedding_path, 'w', encoding='utf-8')
        f.write('%d %d\n' % (len(index2word), embedding_size))
        for word, vec in zip(index2word, vector_strs):
            f.write('%s %s' % (word, vec))

    def print_embedding(self, name='embedding'):
        embeddings = K.batch_get_value([self.model.get_layer(name=name).embeddings])
        for weight in embeddings:
            print(weight)


class CustomEmbedding(Embedding):
    def __init__(self, weights_storeable=True, **kwargs):
        self.weights_storeable = weights_storeable
        super(CustomEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CustomEmbedding, self).build(input_shape)
        if not self.trainable:
            self.non_trainable_weights = self.trainable_weights
            self.trainable_weights = []
        if not self.weights_storeable:
            self.trainable_weights = []
            self.non_trainable_weights = []
        self.built = True


class MaxPoolingMask1D(MaxPooling1D):
    def __init__(self, inputshape, max_value=100000000, **kwargs):
        super(MaxPoolingMask1D, self).__init__(**kwargs)
        self.inputshape = inputshape
        self.max_value = max_value
        assert len(inputshape) == 2

    def build(self, input_shape):
        super(MaxPoolingMask1D, self).build(input_shape)
        self.upper_bound = K.variable(np.zeros(shape=(self.inputshape[0],)) + self.max_value,
                                      name='{}_upper'.format(self.name), dtype='float32')
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        assert mask is not None
        mask = (mask - K.equal(mask, 0)) * self.upper_bound
        mask = K.repeat(mask, self.inputshape[1])
        mask = K.permute_dimensions(mask, (0, 2, 1))
        x = K.minimum(x, mask)
        return super(MaxPoolingMask1D, self).call(x, None)


class AvergeMask1D(Layer):
    def __init__(self, output_dim, **kwargs):
        super(AvergeMask1D, self).__init__(**kwargs)
        self.output_dim = output_dim

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        assert mask is not None
        mask_ave = 1. / K.sum(mask, axis=-1, keepdims=True)
        mask_ave = K.repeat(mask_ave, self.output_dim)
        mask_ave = K.batch_flatten(mask_ave)
        mask_ave = K.cast(mask_ave, 'float32')
        batch_dot = K.batch_dot(mask, x, axes=(1, 1))
        return mask_ave * batch_dot

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 3
        return input_shape[0], self.output_dim


class SumMask1D(Layer):
    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        assert mask is not None
        K.batch_dot(mask, x, axes=(1, 1))

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 3
        return input_shape[0], input_shape[2]


class EmbeddingMaskLayer(Layer):
    def __init__(self, input_size, mask_value=0., is_mask=True, **kwargs):
        self.mask_value = mask_value
        self.is_mask = is_mask
        self.input_size = input_size
        self.Constant = K.variable(np.zeros(input_size), dtype='float32')
        super(EmbeddingMaskLayer, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        if not self.is_mask:
            return None
        return K.any(K.not_equal(input, self.mask_value), axis=-1)

    def call(self, x, mask=None):
        return self.Constant + x

    def get_output_shape_for(self, input_shape):
        return input_shape


class PredictLayer(Layer):
    def __init__( self, input_dim, init='glorot_uniform', activation='linear', weights=None,
                  W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                  W_constraint=None, b_constraint=None,
                  use_bias=True, **kwargs ):

        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.output_dim = 1
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.use_bias = use_bias
        self.initial_weights = weights

        super(PredictLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.init((self.input_dim, self.input_dim),
                           name='{}_W'.format(self.name))
        if self.use_bias:
            self.b = K.zeros((self.output_dim,),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.use_bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.use_bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, inputs, mask=None):
        assert type(inputs) is list
        x, y = inputs
        output = K.sum(x * K.dot(y, self.W), -1, keepdims=True)
        if self.use_bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert type(input_shape) is list
        return input_shape[0][0], self.output_dim


class TensorLayer(Layer):
    def __init__( self, tensor_count, input_dim, init='glorot_uniform', activation='linear',
                  weights=None, W_regularizers=None, V_regularizer=None, b_regularizer=None, activity_regularizer=None,
                  W_constraints=None, V_constranint=None, b_constraint=None, use_v=True, use_bias=True, **kwargs ):
        assert (W_regularizers is None or (
            (type(W_regularizers) is list or type(W_regularizers) is tuple) and len(W_regularizers) == tensor_count
        ))
        assert (W_constraints is None or (
            (type(W_constraints) is list or type(W_constraints) is tuple) and len(W_constraints) == tensor_count
        ))

        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.tensor_count = tensor_count
        self.input_dim = input_dim

        if W_regularizers is None:
            W_regularizers = [None for i in range(tensor_count)]
        self.W_regularizers = [regularizers.get(W_regularizer) for W_regularizer in W_regularizers]
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        if W_constraints is None:
            W_constraints = [None for i in range(tensor_count)]
        self.W_constraints = [constraints.get(W_constraint) for W_constraint in W_constraints]
        self.V_constraint = constraints.get(V_constranint)
        self.b_constraint = constraints.get(b_constraint)

        self.use_v = use_v
        self.use_bias = use_bias

        self.initial_weights = weights

        super(TensorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = [self.init((self.input_dim, self.input_dim),
                            name='{}_W{}'.format(self.name, i)) for i in range(self.tensor_count)]
        self.trainable_weights = []
        for W in self.W:
            self.trainable_weights.append(W)

        if self.use_v:
            self.V = self.init((self.input_dim * 2, self.tensor_count), name='{}_V'.format(self.name))
            self.trainable_weights.append(self.V)

        if self.use_bias:
            self.b = K.zeros((self.tensor_count,),
                             name='{}_b'.format(self.name))
            self.trainable_weights.append(self.b)

        self.regularizers = []
        for W, W_regularizer in zip(self.W, self.W_regularizers):
            if W_regularizer:
                W_regularizer.set_param(W)
                self.regularizers.append(W_regularizer)

        if self.use_v and self.V_regularizer:
            self.V_regularizer.set_param(self.V)
            self.regularizers.append(self.V_regularizer)

        if self.use_bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        for W, W_constraint in zip(self.W, self.W_constraints):
            if W_constraint:
                self.constraints[self.W] = self.W_constraints

        if self.use_v and self.V_constraint:
            self.constraints[self.V] = self.V_constraint

        if self.use_bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, inputs, mask=None):
        assert type(inputs) is list
        x, y = inputs
        if self.tensor_count == 1:
            output = K.sum(x * K.dot(y, self.W[0]), -1, keepdims=True)
        else:
            output = [K.sum(x * K.dot(y, W), -1, keepdims=True) for W in self.W]
            output = K.concatenate(output, axis=-1)
        if self.use_v:
            output += K.dot(K.concatenate(inputs, axis=-1), self.V)
        if self.use_bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert type(input_shape) is list
        return input_shape[0][0], self.tensor_count


class AttentionLayer(Layer):
    """
    the output of this layer is attention weights, not the embedding normed with attention weights
    it's a simple variety of bengio's attention model.
    you can find details in Yang et.'s paper: Hierarchical Attention Networks for Document Classification.
    the input is a seq_embedding, we can compute the output as follow:
            define:
                seq_embedding = [h_1, ... , h_n]
            then:
                a_i = tanh(h_i.dot(W) + b).dot(u)
                norm_a_i = exp(a_i) / sum([exp(a_1), ... , exp(a_n)])
                output = [norm_a_1, ... , norm_a_n]
    """

    def __init__( self, input_dim, inner_output_dim, init='glorot_uniform', inner_activation='tanh',
                  W_regularizer=None, b_regularizer=None, u_regularizer=None,
                  W_constraint=None, b_constraint=None, u_constraint=None,
                  use_bias=True, **kwargs ):
        self.W, self.b, self.u = None, None, None

        self.init = initializers.get(init)
        self.inner_activation = activations.get(inner_activation)
        self.input_dim, self.inner_output_dim = input_dim, inner_output_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.u_constraint = constraints.get(u_constraint)

        self.use_bias = use_bias

        super(AttentionLayer, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return None

    def build(self, input_shape):
        self.W = self.init((self.input_dim, self.inner_output_dim), name='{}_W'.format(self.name))
        self.u = self.init((self.inner_output_dim,), name='{}_u'.format(self.name))
        if self.use_bias:
            self.b = K.zeros((self.inner_output_dim,), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b, self.u]
        else:
            self.trainable_weights = [self.W, self.u]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.use_bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.u_regularizer:
            self.u_regularizer.set_param(self.u)
            self.regularizers.append(self.u_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.use_bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint
        if self.u_constraint:
            self.constraints[self.u] = self.u_constraint

        self.built = True

    def call(self, x, mask=None):
        output = K.dot(x, self.W)
        if self.use_bias:
            output += self.b
        output = self.inner_activation(output)
        output = K.dot(output, self.u)

        if mask is None:
            output = K.softmax(output)
        else:  # softmax with mask
            # output = K.exp(output - K.max(output, axis=-1, keepdims=True)) * mask # method 1
            # output = K.clip(output, -25, 25)  # method 2
            # output = K.exp(output) * K.cast(mask, dtype='float32')
            # output = K.exp(output) * mask
            output = K.exp(output - K.max(output, axis=-1, keepdims=True)) * mask
            output = output / K.sum(output, axis=-1, keepdims=True)

        # # avoid numerical instability with _EPSILON clipping
        # output = K.clip(output, K.epsilon(), 1.0 - K.epsilon())

        return output

    def get_output_shape_for(self, input_shape):
        assert input_shape and (len(input_shape) == 3)
        return (input_shape[0], input_shape[1])

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'inner_output_dim': self.inner_output_dim,
                  'init': self.init.__name__,
                  'activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'u_regularizer': self.u_regularizer.get_config() if self.u_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'u_constraint': self.u_constraint.get_config() if self.u_constraint else None,
                  'bias': self.use_bias}

        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_norm_layer(self, name=None):
        class NormLayer(Layer):
            def __init__(self, **kwargs):
                super(NormLayer, self).__init__(**kwargs)

            def compute_mask(self, input, input_mask=None):
                return None

            def build(self, input_shape):
                self.built = True

            def call(self, x, mask=None):
                assert x and isinstance(x, list) and len(x) == 2
                x_in, attention_in = x
                attention_in = K.expand_dims(attention_in, -1)
                return K.sum(x_in * attention_in, axis=-2, keepdims=False)

            def get_output_shape_for(self, input_shape):
                assert input_shape and len(input_shape) == 2
                x_input_shape = input_shape[0]
                return (x_input_shape[0], x_input_shape[2])

        return NormLayer(name=name)


class AttentionGRU(Recurrent):
    '''
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
    '''

    def __init__(self, output_dim, attention_out_dim=None,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid', attention_activation='tanh',
                 W_regularizer=None, U_regularizer=None, C_regularizer=None, b_regularizer=None,
                 W_a_regularizer=None, U_a_regularizer=None, b_a_regularizer=None, v_a_regularizer=None,
                 dropout_W=0., dropout_U=0., dropout_C=0., **kwargs):
        self.output_dim = output_dim
        self.attention_out_dim = attention_out_dim
        if attention_out_dim is None:
            self.attention_out_dim = output_dim
        self.init = initializers.get(init)
        self.inner_init = initializers.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.attention_activation = activations.get(attention_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.C_regularizer = regularizers.get(C_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_a_regularizer = regularizers.get(W_a_regularizer)
        self.U_a_regularizer = regularizers.get(U_a_regularizer)
        self.b_a_regularizer = regularizers.get(b_a_regularizer)
        self.v_a_regularizer = regularizers.get(v_a_regularizer)
        self.dropout_W, self.dropout_U, self.dropout_C = dropout_W, dropout_U, dropout_C

        if self.dropout_W or self.dropout_U or self.dropout_C:
            self.uses_learning_phase = True
        super(AttentionGRU, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        assert (isinstance(input_shape, list) or isinstance(input_shape, tuple)) and len(input_shape) == 2
        input_shape = input_shape[1]
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    def compute_mask(self, input, mask):
        if self.return_sequences and mask is not None:
            return mask[1]
        else:
            return None

    def build(self, input_shape):
        assert (isinstance(input_shape, list) or isinstance(input_shape, tuple)) and len(input_shape) == 2

        self.input_spec = [InputSpec(shape=spec) for spec in input_shape]

        context_input_dim = input_shape[0][2]
        input_shape = input_shape[1]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        self.W_z = self.init((self.input_dim, self.output_dim),
                             name='{}_W_z'.format(self.name))
        self.U_z = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_z'.format(self.name))
        self.C_z = self.inner_init((context_input_dim, self.output_dim),
                                   name='{}_C_z'.format(self.name))
        self.b_z = K.zeros((self.output_dim,), name='{}_b_z'.format(self.name))

        self.W_r = self.init((self.input_dim, self.output_dim),
                             name='{}_W_r'.format(self.name))
        self.U_r = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_r'.format(self.name))
        self.C_r = self.inner_init((context_input_dim, self.output_dim),
                                   name='{}_C_r'.format(self.name))
        self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))

        self.W_h = self.init((self.input_dim, self.output_dim),
                             name='{}_W_h'.format(self.name))
        self.U_h = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_h'.format(self.name))
        self.C_h = self.inner_init((context_input_dim, self.output_dim),
                                   name='{}_C_h'.format(self.name))
        self.b_h = K.zeros((self.output_dim,), name='{}_b_h'.format(self.name))

        self.W_a = self.init((self.output_dim, self.attention_out_dim),
                             name='{}_W_a'.format(self.name))
        self.U_a = self.inner_init((context_input_dim, self.attention_out_dim),
                                   name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.attention_out_dim,), name='{}_b_a'.format(self.name))
        self.v_a = K.ones((self.attention_out_dim,), name='{}_v_a'.format(self.name))

        self.trainable_weights = [self.W_z, self.U_z, self.C_z, self.b_z,
                                  self.W_r, self.U_r, self.C_r, self.b_r,
                                  self.W_h, self.U_h, self.C_h, self.b_h,
                                  self.W_a, self.U_a, self.b_a, self.v_a]

        self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
        self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
        self.C = K.concatenate([self.C_z, self.C_r, self.C_h])
        self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        self.init_regularizers()

        self.built = True

    def init_regularizers(self):
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.C_regularizer:
            self.C_regularizer.set_param(self.C)
            self.regularizers.append(self.C_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        if self.W_a_regularizer:
            self.W_a_regularizer.set_param(self.W_a)
            self.regularizers.append(self.W_a_regularizer)
        if self.U_a_regularizer:
            self.U_a_regularizer.set_param(self.U_a)
            self.regularizers.append(self.U_a_regularizer)
        if self.b_a_regularizer:
            self.b_a_regularizer.set_param(self.b_a)
            self.regularizers.append(self.b_a_regularizer)
        if self.v_a_regularizer:
            self.v_a_regularizer.set_param(self.v_a)
            self.regularizers.append(self.v_a_regularizer)

    def call(self, x, mask=None, initial_state=None, training=None):
        # input is a list or a tuple and len(input)=2, input[0] is encoder result, input[1] is decoder input
        # input[0] shape(encoder shape): (nb_samples, time (padded with zeros), input_dim)
        # input[1] shape(decoder shape): (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        encode, decode = x
        decode_input_shape = self.input_spec[1].shape
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(decode)

        if mask is None:
            mask = [None, None]

        constants = self.get_constants_with_mask(x, mask)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[1],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=decode_input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_C = states[2]
        encode = states[3]
        mask_encode = states[4]

        timesteps = self.input_spec[0].shape[1]
        attention = K.repeat(K.dot(h_tm1, self.W_a), timesteps) + K.dot(encode, self.U_a) + self.b_a
        attention = K.dot(self.attention_activation(attention), self.v_a)
        if mask_encode is None:
            attention = K.softmax(attention)
        else:  # softmax with mask
            attention = K.exp(attention - K.max(attention, axis=-1, keepdims=True)) * mask_encode
            attention = attention / K.sum(attention, axis=-1, keepdims=True)
        attention = K.expand_dims(attention, -1)
        c = K.sum(encode * attention, axis=-2, keepdims=False)

        x_z = x[:, :self.output_dim]
        x_r = x[:, self.output_dim: 2 * self.output_dim]
        x_h = x[:, 2 * self.output_dim:]

        z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z) + K.dot(c * B_C[0], self.C_z))
        r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r) + K.dot(c * B_C[1], self.C_r))

        hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h) + K.dot(c * B_C[2], self.C_h))
        h = (1 - z) * h_tm1 + z * hh

        return h, [h]

    def get_constants(self, inputs, training=None):
        return self.get_constants_with_mask(inputs, None)

    def get_constants_with_mask(self, x, mask):
        encode, decode = x
        if mask is None:
            mask_e, mask_d = None, None
        else:
            mask_e, mask_d = mask

        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(decode[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_C < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(encode[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, input_dim))
            B_C = [K.in_train_phase(K.dropout(ones, self.dropout_C), ones) for _ in range(3)]
            constants.append(B_C)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        constants.append(encode)

        constants.append(mask_e)
        return constants

    def reset_states(self, states_value=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        if not self.input_spec:
            raise RuntimeError('Layer has never been called '
                               'and thus has no states.')
        input_shape = self.input_spec[1].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x, training=None):
        input_dim = self.input_spec[1].shape[2]
        timesteps = self.input_spec[1].shape[1]

        x = x[1]

        x_z = _time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W,
                                     input_dim, self.output_dim, timesteps)
        x_r = _time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W,
                                     input_dim, self.output_dim, timesteps)
        x_h = _time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
                                     input_dim, self.output_dim, timesteps)
        return K.concatenate([x_z, x_r, x_h], axis=2)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(AttentionGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
