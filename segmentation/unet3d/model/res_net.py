import keras.layers as l
from keras.engine import Model
from keras.optimizers import Adam
from ..metrics import weighted_dice_coefficient_loss


def BnRelu(input):
    bn = l.BatchNormalization(axis=1)(input)
    relu = l.ReLU()(bn)
    return relu


def Res(input):
    relu = BnRelu(input)
    conv = l.Conv3D(64, (1, 3, 3), padding="same")(relu)
    relu = BnRelu(conv)
    conv = l.Conv3D(64, (3, 3, 3), padding="same")(relu)
    add = l.Add()([input, conv])
    return add


def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    input = l.Input(input_shape)

    conv1a = l.Conv3D(32, (3, 3, 3), padding="same")(input)
    relu = BnRelu(conv1a)
    conv1b = l.Conv3D(64, (1, 3, 3), padding="same")(relu)
    relu = BnRelu(conv1b)
    conv1c = l.Conv3D(64, (3, 3, 3), padding="same")(relu)
    relu = BnRelu(conv1c)
    pool = l.MaxPool3D()(relu)
    res2 = Res(pool)
    res3 = Res(res2)
    relu = BnRelu(res3)
    conv4 = l.Conv3D(64, (3, 3, 3), padding="same")(relu)
    pool = l.MaxPool3D()(conv4)
    res5 = Res(pool)
    res6 = Res(res5)
    relu = BnRelu(res6)
    conv7 = l.Conv3D(64, (3, 3, 3), padding="same")(relu)
    pool = l.MaxPool3D()(conv7)
    res8 = Res(pool)
    res9 = Res(res8)

    up1 = conv1b
    up2 = l.UpSampling3D()(res3)
    up3 = l.UpSampling3D()(res6)
    up3 = l.UpSampling3D()(up3)
    up4 = l.UpSampling3D()(res9)
    up4 = l.UpSampling3D()(up4)
    up4 = l.UpSampling3D()(up4)

    add = l.Add()([up1, up2, up3, up4])

    conv = l.Conv3D(n_labels, kernel_size=3, strides=1, padding="same")(add)
    output = l.ReLU()(conv)

    model = Model(inputs=input, outputs=output)
    model.summary()
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model
