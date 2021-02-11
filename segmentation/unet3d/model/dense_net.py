import tensorflow as tf
import keras.layers as l
from keras.engine import Model
from keras.optimizers import Adam
from ..metrics import weighted_dice_coefficient_loss


def conv_block(x, stage, branch, nb_filter, dropout_rate=None):
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = l.BatchNormalization(name=conv_name_base + '_x1_bn', axis=1)(x)
    x = l.Activation('relu', name=relu_name_base + '_x1')(x)
    x = l.Conv3D(inter_channel, kernel_size=1, strides=1, padding="same", name=conv_name_base + '_x1', use_bias=False)(x)

    if dropout_rate:
        x = l.Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = l.BatchNormalization(name=conv_name_base + '_x2_bn', axis=1)(x)
    x = l.Activation('relu', name=relu_name_base + '_x2')(x)
    #x = l.ZeroPadding3D((1, 1, 1), name=conv_name_base + '_x2_zeropadding')(x)
    x = l.Conv3D(nb_filter, kernel_size=3, strides=1, padding="same", name=conv_name_base + '_x2', use_bias=False)(x)

    if dropout_rate:
        x = l.Dropout(dropout_rate)(x)
    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None,
                grow_nb_filters=True):
    concat_feat = x
    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate)
        concat_feat = l.Concatenate(axis=1)([concat_feat, x])

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None):
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = l.BatchNormalization(name=conv_name_base + '_bn', axis=1)(x)
    x = l.Activation('relu', name=relu_name_base)(x)
    x = l.Conv3D(int(nb_filter * compression), kernel_size=1, strides=1, padding="same", name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = l.Dropout(dropout_rate)(x)

    # x = l.AveragePooling3D((2, 2, 2), strides=(2, 2, 2), name=pool_name_base)(x)

    return x


def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    growth_rate = 8
    nb_layers = [6, 12, 24, 16]
    reduction = 0
#    batch_size = 32

    # compute compression factor
    compression = 1.0 - reduction

    nb_dense_block = len(nb_layers)
    # From architecture for ImageNet (Table 1 in the paper)
    # nb_filter = 64
    # nb_layers = [6,12,24,16] # For DenseNet-121

    input = l.Input(shape=input_shape, name='data')

#    x = l.ZeroPadding3D((3, 3, 3), name='conv1_zeropadding', batch_size=batch_size)(input)
    x = l.Conv3D(n_base_filters, kernel_size=5, strides=1, name='conv1', padding="same", use_bias=False)(input)
    x = l.BatchNormalization(name='conv1_bn', axis=1)(x)
    x = l.Activation('relu', name='relu1')(x)
#    x = l.ZeroPadding3D((1, 1, 1), name='pool1_zeropadding')(x)
#    x = l.MaxPooling3D((3, 3, 3), strides=(2, 2, 2), name='pool1')(x)

    stage = 0
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, n_base_filters = dense_block(x, stage, nb_layers[block_idx], n_base_filters, growth_rate,
                                        dropout_rate=dropout_rate)

        # Add transition_block
        x = transition_block(x, stage, n_base_filters, compression=compression, dropout_rate=dropout_rate)
        n_base_filters = int(n_base_filters * compression)

    final_stage = stage + 1
    x, n_base_filters = dense_block(x, final_stage, nb_layers[-1], n_base_filters, growth_rate,
                                    dropout_rate=dropout_rate)

    x = l.Conv3D(n_labels, kernel_size=3, strides=1, padding="same")(x)
    x = l.BatchNormalization(name='conv_final_blk_bn', axis=1)(x)
    output = l.Activation('relu', name='relu_final_blk')(x)

    # x = l.GlobalAveragePooling3D(name='pool_final')(x)
    # x = l.Dense(n_labels, name='fc6')(x)
    # output = l.Activation('softmax', name='prob')(x)

    model = Model(inputs=input, outputs=output)
    model.summary()
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model
