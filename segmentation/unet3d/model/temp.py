import keras.layers as l
from keras.engine import Model


def Conv(input, out_dim):
    conv = l.Conv3D(out_dim, kernel_size=3, strides=1, padding="same")(input)
    bn = l.BatchNormalization()(conv)
    act = l.ReLU()(bn)
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
    concat = l.Concatenate(axis=-1)([x, y])
    return concat


input = l.Input([128, 128, 128, 4])

dim = 16

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

output = l.Conv3D(6, kernel_size=3, strides=1, padding="same", activation=l.LeakyReLU)(conv5)

model = Model(inputs=input, outputs=output)

model.summary()
