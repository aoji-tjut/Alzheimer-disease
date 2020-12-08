from functools import partial
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.engine import Model
from keras.optimizers import Adam
from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss, dice_coefficient_loss

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=4, n_labels=3, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2 ** level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            conv1 = conv(current_layer, n_level_filters, kernel=(1, 1, 1), strides=(1, 1, 1), dropout_rate=dropout_rate)
        else:
            conv1 = conv(current_layer, n_level_filters, kernel=(1, 1, 1), strides=(2, 2, 2), dropout_rate=dropout_rate)

        conv3 = conv(conv1, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1), dropout_rate=dropout_rate)
        summation_layer = Add()([conv1, conv3])

        # conv5 = conv(conv1, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1), dropout_rate=dropout_rate)
        # summation_layer = Add()([conv1, conv5])

        # conv3 = conv(conv1, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1), dropout_rate=dropout_rate)
        # conv3 = conv(conv3, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1), dropout_rate=dropout_rate)
        # summation_layer = Add()([conv1, conv3])

        # conv3 = conv(conv1, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1), dropout_rate=dropout_rate)
        # conv5 = conv(conv1, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1), dropout_rate=dropout_rate)
        # summation_layer = Add()([conv1, conv3, con5])

        # conv3 = conv(conv1, n_level_filters, kernel=(3, 3, 3), strides=(1, 1, 1), dropout_rate=dropout_rate)
        # conv5 = conv(conv1, n_level_filters, kernel=(5, 5, 5), strides=(1, 1, 1), dropout_rate=dropout_rate)
        # concat = concatenate([conv3, conv5], axis=1)
        # conv_concat = conv(concat, n_level_filters, kernel=(1, 1, 1), strides=(1, 1, 1), dropout_rate=dropout_rate)
        # summation_layer_Add()([conv1, conv_concat])

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
