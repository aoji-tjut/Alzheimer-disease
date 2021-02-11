# u-net
import keras.layers as l
from keras.engine import Model
from keras.optimizers import Adam
from ..metrics import dice_coefficient_loss
from ..metrics import weighted_dice_coefficient_loss


def Conv(input, out_dim):
    conv = l.Conv3D(out_dim, kernel_size=3, strides=1, padding="same")(input)
    bn = l.BatchNormalization()(conv)
    act = l.LeakyReLU()(bn)
    conv = l.Conv3D(out_dim, kernel_size=3, strides=1, padding="same")(act)
    bn = l.BatchNormalization()(conv)
    return bn


def Up(input):
    up = l.UpSampling3D()(input)
    # deconv = l.Conv3DTranspose(out_dim, kernel_size=3, strides=1, padding="same",output_padding=(1,1,1)),(input)
    # bn = l.BatchNormalization()(deconv)
    # act = l.ReLU()(bn)
    return up


def Pool(input):
    pool = l.MaxPool3D()(input)
    return pool


def Concat(x, y):
    concat = l.Concatenate(axis=1)([x, y])
    return concat


def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      #loss_function=dice_coefficient_loss, activation_name="sigmoid"):
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    input = l.Input(input_shape)
    dim = n_base_filters

    # Down sampling
    down1 = Conv(input, dim * 1)
    pool1 = Pool(down1)
    down2 = Conv(pool1, dim * 2)
    pool2 = Pool(down2)
    down3 = Conv(pool2, dim * 4)
    pool3 = Pool(down3)
    down4 = Conv(pool3, dim * 8)
    pool4 = Pool(down4)
    down5 = Conv(pool4, dim * 16)
    pool5 = Pool(down5)

    bridge = Conv(pool5, dim * 32)

    # Up sampling
    up1 = Up(bridge)
    concat1 = Concat(up1, down5)
    conv1 = Conv(concat1, dim * 16)
    up2 = Up(conv1)
    concat2 = Concat(up2, down4)
    conv2 = Conv(concat2, dim * 8)
    up3 = Up(conv2)
    concat3 = Concat(up3, down3)
    conv3 = Conv(concat3, dim * 4)
    up4 = Up(conv3)
    concat4 = Concat(up4, down2)
    conv4 = Conv(concat4, dim * 2)
    up5 = Up(conv4)
    concat5 = Concat(up5, down1)
    conv5 = Conv(concat5, dim * 1)

    conv = l.Conv3D(n_labels, kernel_size=1, strides=1, padding="same")(conv5)
    output = l.Activation("sigmoid")(conv)

    model = Model(inputs=input, outputs=output)
    model.summary()
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)

    return model
