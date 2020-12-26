import tensorflow as tf
import keras
from functools import partial
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, Reshape, \
    GlobalAvgPool3D, GlobalMaxPool3D, Activation, ReLU, Lambda, Concatenate, Multiply
from keras import backend as K
from keras.engine import Model
from keras.optimizers import Adam
from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss, dice_coefficient_loss

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def FC(x, channel, r):
    conv = Conv3D(filters=channel // r, kernel_size=1, padding="same", use_bias=False)(x)
    relu = ReLU()(conv)
    out = Conv3D(filters=channel, kernel_size=1, padding="same", use_bias=False)(relu)

    return out


def ChannelAttention(x, channel):
    avg = GlobalAvgPool3D()(x)
    avg = Reshape((avg.shape[1], 1, 1, 1))(avg)
    avg = FC(avg, channel, 4)

    max = GlobalMaxPool3D()(x)
    max = Reshape((max.shape[1], 1, 1, 1))(max)
    max = FC(max, channel, 4)

    add = Add()([avg, max])
    sigmoid = Activation("sigmoid")(add)

    out = Multiply()([x, sigmoid])

    return out


def avg(x):
    return tf.reduce_mean(x, axis=1)


def max(x):
    return tf.reduce_max(x, axis=1)


def SpatialAttention(x):
    avg_mask = keras.layers.Lambda(lambda X: K.mean(X, axis=1, keepdims=True))(x)
    max_mask = keras.layers.Lambda(lambda X: K.max(X, axis=1, keepdims=True))(x)
    mask = Concatenate(axis=1)([avg_mask, max_mask])

    conv = Conv3D(filters=1, kernel_size=7, padding="same", activation="sigmoid")(mask)

    out = Multiply()([x, conv])

    return out


def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=3, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2 ** level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            res = create_convolution_block(current_layer, n_level_filters, kernel=(1, 1, 1), strides=(1, 1, 1))
        else:
            res = create_convolution_block(current_layer, n_level_filters, kernel=(1, 1, 1), strides=(2, 2, 2))

        # conv3 = create_convolution_block(res, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1))
        # conv3 = create_convolution_block(conv3, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1))
        # conv3 = create_convolution_block(conv3, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1))
        # dropout = SpatialDropout3D(rate=dropout_rate, data_format="channels_first")(conv3)
        # summation_layer = Add()([res, dropout])

        # conv5 = create_convolution_block(res, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1))
        # conv5 = create_convolution_block(conv5, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1))
        # conv5 = create_convolution_block(conv5, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1))
        # dropout = SpatialDropout3D(rate=dropout_rate, data_format="channels_first")(conv5)
        # summation_layer = Add()([res, dropout])

        # conv1 = create_convolution_block(res, n_level_filters, kernel=(1, 1, 1), strides=(1, 1, 1))
        # conv1 = create_convolution_block(conv1, n_level_filters, kernel=(1, 1, 1), strides=(1, 1, 1))
        # conv1 = create_convolution_block(conv1, n_level_filters, kernel=(1, 1, 1), strides=(1, 1, 1))
        # dropout_conv1 = SpatialDropout3D(rate=dropout_rate, data_format="channels_first")(conv1)
        # conv3 = create_convolution_block(res, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1))
        # conv3 = create_convolution_block(conv3, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1))
        # conv3 = create_convolution_block(conv3, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1))
        # dropout_conv3 = SpatialDropout3D(rate=dropout_rate, data_format="channels_first")(conv3)
        # conv5 = create_convolution_block(res, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1))
        # conv5 = create_convolution_block(conv5, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1))
        # conv5 = create_convolution_block(conv5, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1))
        # dropout_conv5 = SpatialDropout3D(rate=dropout_rate, data_format="channels_first")(conv5)
        # summation_layer = Add()([res, dropout_conv1, dropout_conv3, dropout_conv5])

        conv1 = create_convolution_block(res, n_level_filters, kernel=(1, 1, 1), strides=(1, 1, 1))
        conv1 = create_convolution_block(conv1, n_level_filters, kernel=(1, 1, 1), strides=(1, 1, 1))
        conv1 = create_convolution_block(conv1, n_level_filters, kernel=(1, 1, 1), strides=(1, 1, 1))
        conv3 = create_convolution_block(res, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1))
        conv3 = create_convolution_block(conv3, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1))
        conv3 = create_convolution_block(conv3, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1))
        conv5 = create_convolution_block(res, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1))
        conv5 = create_convolution_block(conv5, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1))
        conv5 = create_convolution_block(conv5, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1))
        concat = concatenate([conv1, conv3, conv5], axis=1)
        conv_concat = create_convolution_block(concat, n_level_filters, kernel=(1, 1, 1), strides=(1, 1, 1))
        dropout = SpatialDropout3D(rate=dropout_rate, data_format="channels_first")(conv_concat)

        # Attention
        out_channel = ChannelAttention(dropout, n_level_filters)
        out_spatial = SpatialAttention(out_channel)

        # Add
        summation_layer = Add()([res, dropout, out_spatial])

        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    model.summary()
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def conv(input_layer, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1),
         dropout_rate=0.5, data_format="channels_first"):
    conv = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters,
                                    kernel=kernel, strides=strides)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(conv)
    return dropout
