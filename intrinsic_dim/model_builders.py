# Copyright (c) 2018 Uber Technologies, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from IPython import embed
import pdb

import tensorflow as tf
from keras.layers import (Dense, Flatten, Input, Activation, 
                          Reshape, Dropout, Convolution2D, 
                          MaxPooling2D, BatchNormalization, 
                          Conv2D, GlobalAveragePooling2D, 
                          Concatenate, AveragePooling2D, 
                          LocallyConnected2D)
import keras.backend as K

from general.tfutil import hist_summaries_traintest, scalar_summaries_traintest

from keras_ext.engine import ExtendedModel
from keras_ext.layers import (RProjDense, 
                              RProjConv2D, 
                              RProjBatchNormalization, 
                              RProjLocallyConnected2D)
from keras_ext.rproj_layers_util import (OffsetCreatorDenseProj, 
                                         OffsetCreatorSparseProj, 
                                         OffsetCreatorFastfoodProj, 
                                         FastWalshHadamardProjector, 
                                         ThetaPrime)
from keras_ext.util import make_image_input_preproc
from keras.regularizers import l2


def make_and_add_losses(model, input_labels):
    '''Add classification and L2 losses'''

    with tf.name_scope('losses') as scope:
        prob = tf.nn.softmax(model.v.logits, name='prob')
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.v.logits, labels=input_labels, name='cross_ent')
        loss_cross_ent = tf.reduce_mean(cross_ent, name='loss_cross_ent')
        model.add_trackable('loss_cross_ent', loss_cross_ent)
        class_prediction = tf.argmax(prob, 1)

        prediction_correct = tf.equal(class_prediction, input_labels, name='prediction_correct')
        accuracy = tf.reduce_mean(tf.to_float(prediction_correct), name='accuracy')
        model.add_trackable('accuracy', accuracy)
        hist_summaries_traintest(prob, cross_ent)
        scalar_summaries_traintest(accuracy)

        model.add_loss_reg()
        if 'loss_reg' in model.v:
            loss = tf.add_n((
                model.v.loss_cross_ent,
                model.v.loss_reg,
            ), name='loss')
        else:
            loss = model.v.loss_cross_ent
        model.add_trackable('loss', loss)

    nontrackable_fields = ['prob', 'cross_ent', 'class_prediction', 'prediction_correct']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])


def build_model_mnist_fc_dir(weight_decay=0, depth=2, width=100, shift_in=None):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        xx = preproc_images
        xx = Flatten()(xx)
        for _ in range(depth):
            xx = Dense(width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        logits = Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_model_cifar_fc_dir(weight_decay=0, d_rate = 0.0, depth=2, width=100, shift_in=None):
    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'
    

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        xx = input_images
        xx = Flatten()(xx)
        xx = Dropout(d_rate)(xx)

        for _ in range(depth):
            xx = Dense(width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
            xx = Dropout(d_rate)(xx)

        logits = Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_model_mnist_fc(weight_decay=0, vsize=100, depth=2, width=100, shift_in=None, proj_type='sparse'):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    assert proj_type in ('dense', 'sparse')
    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)

        xx = preproc_images
        xx = Flatten()(xx)
        for _ in range(depth):
            xx = RProjDense(offset_creator_class, vv, width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        logits = RProjDense(offset_creator_class, vv, 10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)

        model.add_extra_trainable_weight(vv.var)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_model_mnist_fc_fastfood(weight_decay=0, vsize=100, depth=2, width=100, shift_in=None, DD=None):
    '''If DD is not specified, it will be computed.'''

    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, DenseLayer, Conv2DLayer):
        xx = Flatten()(preproc_images)
        for _ in range(depth):
            xx = DenseLayer(width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        logits = DenseLayer(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)
        nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    # Allow manual specification of D if desired
    if not DD:
        # Silly-but-not-too-bad hack to determine direct parameter
        # dimension D: make the whole model using direct layers, then
        # instantiate FastWalshHadamardProjector, then make whole new
        # model that gets its params from FastWalshHadamardProjector
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, Dense, Conv2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    # vsize = DD

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, DenseLayer, Conv2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_model_cifar_fc_fastfood(weight_decay=0, d_rate = 0.0, vsize=100, depth=2, width=100, shift_in=None, DD=None):
    '''If DD is not specified, it will be computed.'''

    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'
    

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, DenseLayer, Conv2DLayer):
        xx = Flatten()(input_images)
        xx = Dropout(d_rate)(xx)
        for _ in range(depth):
            xx = DenseLayer(width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
            xx = Dropout(d_rate)(xx)

        logits = DenseLayer(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)
        nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, Dense, Conv2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, DenseLayer, Conv2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_model_cifar_fc(weight_decay=0, d_rate = 0.0, vsize=100, depth=2, width=100, shift_in=None, proj_type='sparse'):
    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'
    

    assert proj_type in ('dense', 'sparse')
    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)

        xx = preproc_images
        xx = Flatten()(xx)
        xx = Dropout(d_rate)(xx)
        for _ in range(depth):
            xx = RProjDense(offset_creator_class, vv, width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
            xx = Dropout(d_rate)(xx)
        logits = RProjDense(offset_creator_class, vv, 10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)

        model.add_extra_trainable_weight(vv.var)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_cnn_model_direct_mnist(weight_decay=0, shift_in=None):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        xx = Convolution2D(16, kernel_size=3, strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = Convolution2D(16, 3, 3, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Convolution2D(16, 3, 3, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = BatchNormalization(momentum=0.5)(xx)
        xx = Convolution2D(16, 3, 3, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)  # (8, 8)
        xx = MaxPooling2D((2, 2))(xx)  # (4, 4)
        xx = Flatten()(xx)
        xx = Dense(800, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dense(800, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = BatchNormalization(momentum=0.5)(xx)
        xx = Activation('relu')(xx)
        xx = Dense(500, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_cnn_model_mnist(weight_decay=0, vsize=100, shift_in=None):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)

        xx = RProjConv2D(vv, 16, kernel_size=3, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = RProjConv2D(vv, 16,  kernel_size=3, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)

        xx = RProjConv2D(vv, 16, kernel_size=3, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = RProjBatchNormalization(vv_theta=vv, momentum=0.5)(xx)
        xx = RProjConv2D(vv, 16,  kernel_size=3, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)  # (8, 8)
        xx = MaxPooling2D((2, 2))(xx)  # (4, 4)
        xx = Flatten()(xx)
        xx = RProjDense(vv, 800, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = RProjDense(vv, 800, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = RProjBatchNormalization(vv_theta=vv, momentum=0.5)(xx)
        xx = Activation('relu')(xx)
        xx = RProjDense(vv, 500, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = RProjDense(vv, 10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)

        model.add_extra_trainable_weight(vv.var)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model



def build_LeNet_direct_mnist(weight_decay=0, shift_in=None, c1=6, c2=16, d1=120, d2=84):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        xx = Convolution2D(c1, kernel_size=5, strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Convolution2D(c2, kernel_size=5, strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = Dense(d1, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dense(d2, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_UntiedLeNet_direct_mnist(weight_decay=0, shift_in=None, c1=6, c2=16, d1=120, d2=84):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        xx = LocallyConnected2D(c1, kernel_size=5, strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = LocallyConnected2D(c2, kernel_size=5, strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = Dense(d1, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dense(d2, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_MLPLeNet_direct_mnist(weight_decay=0, shift_in=None, c1=6, c2=16, d1=120, d2=84):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    # Obtain the output shape of each layer in LeNet
    M0 = build_LeNet_direct_mnist(weight_decay=weight_decay)  
    inp_ = M0.input                                           # input placeholder
    outputs_ = [layer.output for layer in M0.layers]          # all layer outputs
    functors = [K.function([inp_]+ [K.learning_phase()], [out]) for out in outputs_]  # evaluation functions

    test = np.random.random(im_shape)[np.newaxis,...]
    layer_outs = [func([test, 1.]) for func in functors]
    layer_outs_shape = [l.shape for ll in layer_outs for l in ll]

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        xx = preproc_images
        xx = Flatten()(xx)
        xx = Dense(np.prod(layer_outs_shape[1]), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = Reshape(layer_outs_shape[1][1:])(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = Dense(np.prod(layer_outs_shape[3]), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = Reshape(layer_outs_shape[3][1:])(xx)
        xx = MaxPooling2D((2, 2))(xx) 
        xx = Flatten()(xx)
        xx = Dense(d1, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dense(d2, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)

        model = ExtendedModel(input=input_images, output=logits)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_MLPLeNet_direct_cifar(weight_decay=0, shift_in=None, c1=6, c2=16, d1=120, d2=84):
    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'

    # Obtain the output shape of each layer in LeNet
    M0 = build_LeNet_direct_cifar(weight_decay=weight_decay)  
    inp_ = M0.input                                           # input placeholder
    outputs_ = [layer.output for layer in M0.layers]          # all layer outputs
    functors = [K.function([inp_]+ [K.learning_phase()], [out]) for out in outputs_]  # evaluation functions

    test = np.random.random(im_shape)[np.newaxis,...]
    layer_outs = [func([test, 1.]) for func in functors]
    layer_outs_shape = [l.shape for ll in layer_outs for l in ll]

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        xx = preproc_images
        xx = Flatten()(xx)
        xx = Dense(np.prod(layer_outs_shape[1]), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = Reshape(layer_outs_shape[1][1:])(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = Dense(np.prod(layer_outs_shape[3]), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = Reshape(layer_outs_shape[3][1:])(xx)
        xx = MaxPooling2D((2, 2))(xx) 
        xx = Flatten()(xx)
        xx = Dense(d1, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dense(d2, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)

        model = ExtendedModel(input=input_images, output=logits)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model




def build_LeNet_mnist(weight_decay=0, vsize=100, shift_in=None, proj_type='sparse'):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    assert proj_type in ('dense', 'sparse')
    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj


    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)
        xx = RProjConv2D(offset_creator_class, vv, 6, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = RProjConv2D(offset_creator_class, vv, 16, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = RProjDense(offset_creator_class, vv, 120, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = RProjDense(offset_creator_class, vv, 84, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = RProjDense(offset_creator_class, vv, 10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)
        model.add_extra_trainable_weight(vv.var)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)
    return model



def build_model_mnist_LeNet_fastfood(weight_decay=0, vsize=100, shift_in=None, DD=None):
    '''If DD is not specified, it will be computed.'''

    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, DenseLayer, Conv2DLayer):
        vv = ThetaPrime(vsize)
        xx = Conv2DLayer(6, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Conv2DLayer(16, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = DenseLayer(120, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = DenseLayer(84, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = DenseLayer(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)
        nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, Dense, Conv2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, DenseLayer, Conv2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_model_cifar_LeNet_fastfood(weight_decay=0, vsize=100, shift_in=None, DD=None, d_rate=0.0, c1=6, c2=16, d1=120, d2=84):
    '''If DD is not specified, it will be computed.'''

    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, DenseLayer, Conv2DLayer):
        vv = ThetaPrime(vsize)
        xx = Conv2DLayer(c1, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Conv2DLayer(c2, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = Dropout(d_rate)(xx)
        xx = DenseLayer(d1, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dropout(d_rate)(xx)
        xx = DenseLayer(d2, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dropout(d_rate)(xx)
        logits = DenseLayer(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)
        nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, Dense, Conv2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, DenseLayer, Conv2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model



def build_model_mnist_UntiedLeNet_fastfood(weight_decay=0, vsize=100, shift_in=None, DD=None):
    '''If DD is not specified, it will be computed.'''

    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, DenseLayer, Conv2DLayer, LocallyConnected2DLayer):
        xx = LocallyConnected2DLayer(6, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = LocallyConnected2DLayer(16, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = DenseLayer(120, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = DenseLayer(84, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = DenseLayer(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)
        nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, Dense, Conv2D, LocallyConnected2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        LocallyConnected2DLayer = lambda *args, **kwargs: RProjLocallyConnected2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, DenseLayer, Conv2DLayer, LocallyConnected2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model

def build_model_cifar_UntiedLeNet_fastfood(weight_decay=0, vsize=100, shift_in=None, DD=None):
    '''If DD is not specified, it will be computed.'''

    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, DenseLayer, Conv2DLayer, LocallyConnected2DLayer):
        xx = LocallyConnected2DLayer(6, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = LocallyConnected2DLayer(16, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = DenseLayer(120, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = DenseLayer(84, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = DenseLayer(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)
        nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    # Allow manual specification of D if desired
    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, Dense, Conv2D, LocallyConnected2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        LocallyConnected2DLayer = lambda *args, **kwargs: RProjLocallyConnected2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, DenseLayer, Conv2DLayer, LocallyConnected2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model



def build_model_mnist_MLPLeNet_fastfood(weight_decay=0, vsize=100, shift_in=None, DD=None):
    '''If DD is not specified, it will be computed.'''

    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    # Obtain the output shape of each layer in LeNet
    M0 = build_LeNet_direct_mnist(weight_decay=weight_decay)  
    inp_ = M0.input                                           # input placeholder
    outputs_ = [layer.output for layer in M0.layers]          # all layer outputs
    functors = [K.function([inp_]+ [K.learning_phase()], [out]) for out in outputs_]  # evaluation functions

    test = np.random.random(im_shape)[np.newaxis,...]
    layer_outs = [func([test, 1.]) for func in functors]
    layer_outs_shape = [l.shape for ll in layer_outs for l in ll]

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, DenseLayer, Conv2DLayer):
        xx = preproc_images
        xx = Flatten()(xx)
        xx = DenseLayer(np.prod(layer_outs_shape[1]), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = Reshape(layer_outs_shape[1][1:])(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = DenseLayer(np.prod(layer_outs_shape[3]), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = Reshape(layer_outs_shape[3][1:])(xx)
        xx = MaxPooling2D((2, 2))(xx) 
        xx = Flatten()(xx)
        xx = DenseLayer(120, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = DenseLayer(84, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = DenseLayer(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)

        model = ExtendedModel(input=input_images, output=logits)
        nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, Dense, Conv2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, DenseLayer, Conv2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model      


def build_model_cifar_MLPLeNet_fastfood(weight_decay=0, vsize=100, shift_in=None, DD=None):
    '''If DD is not specified, it will be computed.'''

    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'

    # Obtain the output shape of each layer in LeNet
    M0 = build_LeNet_direct_cifar(weight_decay=weight_decay)  
    inp_ = M0.input                                           # input placeholder
    outputs_ = [layer.output for layer in M0.layers]          # all layer outputs
    functors = [K.function([inp_]+ [K.learning_phase()], [out]) for out in outputs_]  # evaluation functions

    test = np.random.random(im_shape)[np.newaxis,...]
    layer_outs = [func([test, 1.]) for func in functors]
    layer_outs_shape = [l.shape for ll in layer_outs for l in ll]

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, DenseLayer, Conv2DLayer):
        xx = preproc_images
        xx = Flatten()(xx)
        xx = DenseLayer(np.prod(layer_outs_shape[1]), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = Reshape(layer_outs_shape[1][1:])(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = DenseLayer(np.prod(layer_outs_shape[3]), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = Reshape(layer_outs_shape[3][1:])(xx)
        xx = MaxPooling2D((2, 2))(xx) 
        xx = Flatten()(xx)
        xx = DenseLayer(120, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = DenseLayer(84, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = DenseLayer(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)

        model = ExtendedModel(input=input_images, output=logits)
        nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, Dense, Conv2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, DenseLayer, Conv2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model    


def build_LeNet_direct_cifar(weight_decay=0, d_rate=0.0, shift_in=None, c1=6, c2=16, d1=120, d2=84):
    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        xx = Convolution2D(c1, kernel_size=5, strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Convolution2D(c2, kernel_size=5, strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = Dropout(d_rate)(xx)
        xx = Dense(d1, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dropout(d_rate)(xx)
        xx = Dense(d2, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dropout(d_rate)(xx)
        logits = Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_UntiedLeNet_direct_cifar(weight_decay=0, d_rate=0.0, shift_in=None):
    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        xx = LocallyConnected2D(6, kernel_size=5, strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = LocallyConnected2D(16, kernel_size=5, strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = Dropout(d_rate)(xx)
        xx = Dense(120, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dropout(d_rate)(xx)
        xx = Dense(84, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dropout(d_rate)(xx)
        logits = Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_LeNet_cifar(weight_decay=0, vsize=100, shift_in=None, proj_type='sparse'):
    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'

    assert proj_type in ('dense', 'sparse')
    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj


    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)
        xx = RProjConv2D(offset_creator_class, vv, 6, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = RProjConv2D(offset_creator_class, vv, 16, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = Dropout(0.5)(xx)
        xx = RProjDense(offset_creator_class, vv, 120, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dropout(0.5)(xx)
        xx = RProjDense(offset_creator_class, vv, 84, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dropout(0.5)(xx)
        logits = RProjDense(offset_creator_class, vv, 10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)
        model.add_extra_trainable_weight(vv.var)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)
    return model


def build_alexnet_direct(weight_decay=0, shift_in=None):
    '''Build and return an AlexNet model'''

    im_shape = (227,227,3)
    n_label_vals = 1000
    im_dtype = 'uint8'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net'):
        which = 'full_bn'
        if which == '3bn':
            # Alexnet with 3 BN
            xx = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Conv2D(256, (5, 5), padding='same', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BatchNormalization()(xx)
            xx = Activation('relu')(xx)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)  # MOD
            xx = Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
            xx = Conv2D(256, (3, 3), padding='same', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BatchNormalization()(xx)
            xx = Activation('relu')(xx)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Flatten()(xx)
            xx = Dense(4096, kernel_initializer='he_normal', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BatchNormalization()(xx)
            xx = Activation('relu')(xx)
            xx = Dense(4096, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)

        elif which == 'full_bn':
            # Alexnet with BN everywhere
            xx = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='linear', kernel_regularizer=l2(weight_decay))(preproc_images)
            xx = BatchNormalization()(xx)
            xx = Activation('relu')(xx)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Conv2D(256, (5, 5), padding='same', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BatchNormalization()(xx)
            xx = Activation('relu')(xx)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)  # MOD
            xx = BatchNormalization()(xx)
            xx = Activation('relu')(xx)
            xx = Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)  # MOD
            xx = BatchNormalization()(xx)
            xx = Activation('relu')(xx)
            xx = Conv2D(256, (3, 3), padding='same', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BatchNormalization()(xx)
            xx = Activation('relu')(xx)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Flatten()(xx)
            xx = Dense(4096, kernel_initializer='he_normal', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BatchNormalization()(xx)
            xx = Activation('relu')(xx)
            xx = Dropout(0.5)(xx)
            xx = Dense(4096, kernel_initializer='he_normal', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BatchNormalization()(xx)
            xx = Activation('relu')(xx)
            xx = Dropout(0.5)(xx)
        else:
            raise Exception()

        logits = Dense(n_label_vals, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)

    model = ExtendedModel(input=input_images, output=logits)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_alexnet_fastfood(weight_decay=0, vsize=100, shift_in=None, DD=None):
    '''If DD is not specified, it will be computed.'''

    im_shape = (227,227,3)
    n_label_vals = 1000
    im_dtype = 'uint8'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, preproc_images, DenseLayer, Conv2DLayer, BNLayer):
        which = 'full_bn'
        if which == '3bn':
            # Alexnet with 3 BN
            xx = Conv2DLayer(96, (11, 11), strides=(4, 4), padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Conv2DLayer(256, (5, 5), padding='same', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BNLayer()(xx)
            xx = Activation('relu')(xx)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Conv2DLayer(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)  # MOD
            xx = Conv2DLayer(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
            xx = Conv2DLayer(256, (3, 3), padding='same', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BNLayer()(xx)
            xx = Activation('relu')(xx)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Flatten()(xx)
            xx = DenseLayer(4096, kernel_initializer='he_normal', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BNLayer()(xx)
            xx = Activation('relu')(xx)
            xx = DenseLayer(4096, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        elif which == 'full_bn':
            # Alexnet with BN everywhere
            xx = Conv2DLayer(96, (11, 11), strides=(4, 4), padding='valid', activation='linear', kernel_regularizer=l2(weight_decay))(preproc_images)
            xx = BNLayer()(xx)
            xx = Activation('relu')(xx)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Conv2DLayer(256, (5, 5), padding='same', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BNLayer()(xx)
            xx = Activation('relu')(xx)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Conv2DLayer(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)  # MOD
            xx = BNLayer()(xx)
            xx = Activation('relu')(xx)
            xx = Conv2DLayer(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)  # MOD
            xx = BNLayer()(xx)
            xx = Activation('relu')(xx)
            xx = Conv2DLayer(256, (3, 3), padding='same', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BNLayer()(xx)
            xx = Activation('relu')(xx)
            xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)
            xx = Flatten()(xx)
            xx = DenseLayer(4096, kernel_initializer='he_normal', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BNLayer()(xx)
            xx = Activation('relu')(xx)
            xx = Dropout(0.5)(xx)
            xx = DenseLayer(4096, kernel_initializer='he_normal', activation='linear', kernel_regularizer=l2(weight_decay))(xx)
            xx = BNLayer()(xx)
            xx = Activation('relu')(xx)
            xx = Dropout(0.5)(xx)
        else:
            raise Exception()

        logits = DenseLayer(n_label_vals, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    # Allow manual specification of D if desired
    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, preproc_images, Dense, Conv2D, BatchNormalization)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        BNLayer = lambda *args, **kwargs: RProjBatchNormalization(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, preproc_images, DenseLayer, Conv2DLayer, BNLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_squeezenet_direct(weight_decay=0, shift_in=None):
    '''Build and return an AlexNet model'''

    im_shape = (224, 224, 3)
    n_label_vals = 1000
    im_dtype = 'uint8'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net'):
        xx = Conv2D(96, (7, 7), strides=(2, 2), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)

        xx_sqz = Conv2D(16, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx)
        xx_exp1 = Conv2D(64, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2D(64, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx_sqz = Conv2D(16, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_merg)
        xx_exp1 = Conv2D(64, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2D(64, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx_sqz = Conv2D(32, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_merg)
        xx_exp1 = Conv2D(128, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2D(128, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])
        xx = MaxPooling2D((3, 3), strides=(2, 2))(xx_merg)

        xx_sqz = Conv2D(32, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx)
        xx_exp1 = Conv2D(128, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2D(128, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx_sqz = Conv2D(48, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_merg)
        xx_exp1 = Conv2D(192, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2D(192, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx_sqz = Conv2D(48, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_merg)
        xx_exp1 = Conv2D(192, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2D(192, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx_sqz = Conv2D(64, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_merg)
        xx_exp1 = Conv2D(256, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2D(256, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])
        xx = MaxPooling2D((3, 3), strides=(2, 2))(xx_merg)

        xx_sqz = Conv2D(64, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx)
        xx_exp1 = Conv2D(256, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2D(256, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx = Dropout(0.5)(xx_merg)

        xx = Conv2D(n_label_vals, (1, 1), padding='valid',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(weight_decay))(xx)
        xx = AveragePooling2D((13, 13))(xx)
        logits = Flatten()(xx)

    model = ExtendedModel(input=input_images, output=logits)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_squeezenet_fastfood(weight_decay=0, vsize=100, shift_in=None, DD=None):
    '''If DD is not specified, it will be computed.'''

    im_shape = (224, 224, 3)
    n_label_vals = 1000
    im_dtype = 'uint8'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, preproc_images, Conv2DLayer):
        xx = Conv2DLayer(96, (7, 7), strides=(2, 2), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((3, 3), strides=(2, 2))(xx)

        xx_sqz = Conv2DLayer(16, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx)
        xx_exp1 = Conv2DLayer(64, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2DLayer(64, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx_sqz = Conv2DLayer(16, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_merg)
        xx_exp1 = Conv2DLayer(64, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2DLayer(64, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx_sqz = Conv2DLayer(32, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_merg)
        xx_exp1 = Conv2DLayer(128, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2DLayer(128, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])
        xx = MaxPooling2D((3, 3), strides=(2, 2))(xx_merg)

        xx_sqz = Conv2DLayer(32, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx)
        xx_exp1 = Conv2DLayer(128, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2DLayer(128, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx_sqz = Conv2DLayer(48, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_merg)
        xx_exp1 = Conv2DLayer(192, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2DLayer(192, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx_sqz = Conv2DLayer(48, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_merg)
        xx_exp1 = Conv2DLayer(192, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2DLayer(192, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx_sqz = Conv2DLayer(64, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_merg)
        xx_exp1 = Conv2DLayer(256, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2DLayer(256, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])
        xx = MaxPooling2D((3, 3), strides=(2, 2))(xx_merg)

        xx_sqz = Conv2DLayer(64, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx)
        xx_exp1 = Conv2DLayer(256, (1, 1), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_exp2 = Conv2DLayer(256, (3, 3), padding='same',  
                kernel_initializer='glorot_uniform', activation='relu',
                kernel_regularizer=l2(weight_decay))(xx_sqz)
        xx_merg = Concatenate()([xx_exp1,xx_exp2])

        xx = Dropout(0.5)(xx_merg)

        xx = Conv2DLayer(n_label_vals, (1, 1), padding='valid',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(weight_decay))(xx)
        xx = AveragePooling2D((13, 13))(xx)
        logits = Flatten()(xx)

        model = ExtendedModel(input=input_images, output=logits)

        for field in ['logits']:
            model.add_var(field, locals()[field])
            
        return model


    # Allow manual specification of D if desired
    if not DD:
        # Silly-but-not-too-bad hack to determine direct parameter
        # dimension D: make the whole model using direct layers, then
        # instantiate FastWalshHadamardProjector, then make whole new
        # model that gets its params from FastWalshHadamardProjector
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, preproc_images, Conv2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, preproc_images, Conv2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model

