# v-net
import keras
from keras.engine import Model
from keras.optimizers import Adam
from ..metrics import dice_coefficient_loss
from ..metrics import weighted_dice_coefficient_loss


# Building blocks
def adding_conv(x, a, filters, kernel_size, padding, strides, data_format, groups):
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides,
                            activation=None, data_format=data_format)(x)
    c = keras.layers.add([c, a])
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c


def conv(x, filters, kernel_size, padding, strides, data_format, groups):
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides,
                            activation=None, data_format=data_format)(x)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c


def down_conv(x, filters, kernel_size, padding, data_format, groups):
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=2,
                            activation=None, data_format=data_format)(x)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c


def up_conv_concat_conv(x, skip, filters, kernel_size, padding, strides, data_format, groups):
    channel_axis = -1 if data_format == 'channels_last' else 1
    c = keras.layers.Conv3DTranspose(filters, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                                     data_format=data_format)(x)  # up dim(x) by x2
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides,
                            activation=None, data_format=data_format)(c)
    concat = keras.layers.Concatenate(axis=channel_axis)([c, skip])  # concat after Up; dim(skip) == 2*dim(x)
    c = keras.layers.BatchNormalization()(concat)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c


# Encoders
def encoder1(x, filters, kernel_size, padding, strides, data_format, groups):
    conv1 = conv(x, filters, kernel_size, padding, strides, data_format, groups)
    addconv = adding_conv(conv1, conv1, filters, kernel_size, padding, strides, data_format, groups)  # N
    downconv = down_conv(addconv, filters * 2, kernel_size, padding, data_format, groups)  # N/2
    return (addconv, downconv)


def encoder2(x, filters, kernel_size, padding, strides, data_format, groups):
    conv1 = conv(x, filters, kernel_size, padding, strides, data_format, groups)
    addconv = adding_conv(conv1, x, filters, kernel_size, padding, strides, data_format, groups)  # N/2
    downconv = down_conv(addconv, filters * 2, kernel_size, padding, data_format, groups)  # N/4
    return (addconv, downconv)


def encoder3(x, filters, kernel_size, padding, strides, data_format, groups):
    conv1 = conv(x, filters, kernel_size, padding, strides, data_format, groups)  # N/4
    conv2 = conv(conv1, filters, kernel_size, padding, strides, data_format, groups)  # N/4
    addconv = adding_conv(conv2, x, filters, kernel_size, padding, strides, data_format, groups)  # N/4
    downconv = down_conv(addconv, filters * 2, kernel_size, padding, data_format, groups)  # N/8
    return (addconv, downconv)


def encoder4(x, filters, kernel_size, padding, strides, data_format, groups):
    conv1 = conv(x, filters, kernel_size, padding, strides, data_format, groups)  # N/8
    conv2 = conv(conv1, filters, kernel_size, padding, strides, data_format, groups)  # N/8
    addconv = adding_conv(conv2, x, filters, kernel_size, padding, strides, data_format, groups)  # N/8
    downconv = down_conv(addconv, filters * 2, kernel_size, padding, data_format, groups)  # N/16
    return (addconv, downconv)


# Bottom
def bottom(x, filters, kernel_size, padding, strides, data_format, groups):
    conv1 = conv(x, filters, kernel_size, padding, strides, data_format, groups)
    conv2 = conv(conv1, filters, kernel_size, padding, strides, data_format, groups)
    addconv = adding_conv(conv2, x, filters, kernel_size, padding, strides, data_format, groups)  # N/16
    return addconv  # N/16


# Decoders
def decoder4(x, skip, filters, kernel_size, padding, strides, data_format, groups):
    upconv = up_conv_concat_conv(x, skip, filters, kernel_size, padding, strides, data_format, groups)  # N/8
    conv1 = conv(upconv, filters, kernel_size, padding, strides, data_format, groups)
    conv2 = conv(conv1, filters, kernel_size, padding, strides, data_format, groups)
    return conv2  # N/8


def decoder3(x, skip, filters, kernel_size, padding, strides, data_format, groups):
    upconv = up_conv_concat_conv(x, skip, filters, kernel_size, padding, strides, data_format, groups)  # N/4
    conv1 = conv(upconv, filters, kernel_size, padding, strides, data_format, groups)
    conv2 = conv(conv1, filters, kernel_size, padding, strides, data_format, groups)
    return conv2  # N/4


def decoder2(x, skip, filters, kernel_size, padding, strides, data_format, groups):
    upconv = up_conv_concat_conv(x, skip, filters, kernel_size, padding, strides, data_format, groups)  # N/2
    conv1 = conv(upconv, filters, kernel_size, padding, strides, data_format, groups)
    return conv1  # N/2


def decoder1(x, skip, filters, kernel_size, padding, strides, data_format, groups):
    upconv = up_conv_concat_conv(x, skip, filters, kernel_size, padding, strides, data_format, groups)  # N
    return upconv  # N


# Attention gate
def attention_gate(inp, g, intra_filters):
    data_format = 'channels_first'  ##@##
    groups = 8  ##@##

    # Gating signal processing
    g = keras.layers.Conv3D(intra_filters, kernel_size=1, data_format=data_format)(g)  # N/2
    g = keras.layers.BatchNormalization()(g)

    # Skip signal processing:
    x = keras.layers.Conv3D(intra_filters, kernel_size=2, strides=2, padding='same', data_format=data_format)(
        inp)  # N-->N/2
    x = keras.layers.BatchNormalization()(x)  # N

    # Add and proc
    g_x = keras.layers.Add()([g, x])  # N/2
    psi = keras.layers.Activation('relu')(g_x)  # N/2
    psi = keras.layers.Conv3D(1, kernel_size=1, padding='same', data_format=data_format)(psi)  # N/2
    psi = keras.layers.BatchNormalization()(psi)  # N/2
    psi = keras.layers.Activation('sigmoid')(psi)  # N/2
    alpha = keras.layers.UpSampling3D(size=2, data_format=data_format)(psi)  # N/2-->N

    x_hat = keras.layers.Multiply()([inp, alpha])
    return x_hat


# Model
def VNet(n_classes,input_shape, filters, kernel_size, padding, strides, data_format, groups, inter_filters):

        inputs = keras.layers.Input(input_shape)

        (encoder1_addconv, encoder1_downconv) = encoder1(inputs, filters * 2 ** 0, kernel_size, padding, strides,
                                                         data_format, groups)  # N, N/2
        (encoder2_addconv, encoder2_downconv) = encoder2(encoder1_downconv, filters * 2 ** 1, kernel_size, padding,
                                                         strides, data_format, groups)  # N/2, N/4
        (encoder3_addconv, encoder3_downconv) = encoder3(encoder2_downconv, filters * 2 ** 2, kernel_size, padding,
                                                         strides, data_format, groups)  # N/4, N/8
        (encoder4_addconv, encoder4_downconv) = encoder4(encoder3_downconv, filters * 2 ** 3, kernel_size, padding,
                                                         strides, data_format, groups)  # N/8, N/16

        bottom_addconv = bottom(encoder4_downconv, filters * 2 ** 4, kernel_size, padding, strides, data_format,
                                groups)  # N/16

        encoder4_ag = attention_gate(encoder4_addconv, bottom_addconv, inter_filters)  # (N/8, N/16) --> N/8
        decoder4_conv = decoder4(bottom_addconv, encoder4_ag, filters * 2 ** 3, kernel_size, padding, strides,
                                 data_format, groups)  # N/8
        encoder3_ag = attention_gate(encoder3_addconv, decoder4_conv, inter_filters)  # (N/4, N/8) --> N/4
        decoder3_conv = decoder3(decoder4_conv, encoder3_ag, filters * 2 ** 2, kernel_size, padding, strides,
                                 data_format, groups)  # N/4
        encoder2_ag = attention_gate(encoder2_addconv, decoder3_conv, inter_filters)  # (N/2, N/4) --> N/2
        decoder2_conv = decoder2(decoder3_conv, encoder2_ag, filters * 2 ** 1, kernel_size, padding, strides,
                                 data_format, groups)  # N/2
        encoder1_ag = attention_gate(encoder1_addconv, decoder2_conv, inter_filters)  # (N, N/2) --> N
        decoder1_conv = decoder1(decoder2_conv, encoder1_ag, filters * 2 ** 0, kernel_size, padding, strides,
                                 data_format, groups)  # N

        outputs = keras.layers.Conv3D(n_classes,
                                      (1, 1, 1),
                                      padding='same',
                                      activation='sigmoid',
                                      data_format=data_format)(decoder1_conv)

        model = Model(input=inputs, output=outputs)

        return model

def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    model = VNet(n_labels,input_shape,n_base_filters,2,"same",1,"channels_first",0,8)
    model.summary()
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)

    return model
