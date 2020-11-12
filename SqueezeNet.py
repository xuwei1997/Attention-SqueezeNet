import h5py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, concatenate
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.utils import plot_model


# #keras2 中add应该改为 add.add
def SqueezeNet(nb_classes,inputs=(224, 224,3)):  #此处的输入可能有错，默认输入应该为(batch, height, width, channels)
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)

    Arguments:
        nb_classes: total number of final categories
        
        inputs -- shape of the input images (channel, cols, rows)
    """

    input_img = Input(shape=inputs)
    conv1 = Convolution2D(
        96,
        (7,7),
        activation='relu',
        # init='glorot_uniform',
        # subsample=(2, 2),
        strides= (2, 2) ,
        padding='same',
        data_format='channels_last',
        name='conv1')(input_img)
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            name='maxpool1')(conv1)

    fire2_squeeze = Convolution2D(
        16,
        (1,1),
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire2_squeeze')(maxpool1)
    fire2_expand1 = Convolution2D(
        64,
        (1,1),
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire2_expand1')(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64,
        (3,3),
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire2_expand2')(fire2_squeeze)

    print(fire2_expand1.shape)
    print(fire2_expand2.shape)
    add2 = concatenate([fire2_expand1, fire2_expand2],axis=1)
    # mode='concat',
    # concat_axis=1)

    fire3_squeeze = Convolution2D(
        16,
        (1,1),
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire3_squeeze')(add2)
    #    fire3_expand1 = Convolution2D(64,
    #                                   1,
    #                                   1,
    #                                   activation='relu',
    #                                   # init='glorot_uniform',
    #                                   padding = 'same',
    #                                   name='fire3_expand1')(fire3_squeeze)
    fire3_expand1 = Convolution2D(
        64,
        (1,1),
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire3_expand1')(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64,
        (3,3),
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire3_expand2')(fire3_squeeze)
    add3 = concatenate([fire3_expand1, fire3_expand2],axis=1)
    # mode='concat',
    # concat_axis=1)

    fire4_squeeze = Convolution2D(
        32,
        (1,1),
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire4_squeeze')(add3)
    fire4_expand1 = Convolution2D(
        128,
        (1,1),
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire4_expand1')(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128,
        (3,3),
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire4_expand2')(fire4_squeeze)
    add4 = concatenate([fire4_expand1, fire4_expand2],axis=1)
    # mode='concat',
    # concat_axis=1)
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            name='maxpool4')(add4)

    fire5_squeeze = Convolution2D(
        32,
        1,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire5_squeeze')(maxpool4)
    fire5_expand1 = Convolution2D(
        128,
        1,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire5_expand1')(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        128,
        3,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire5_expand2')(fire5_squeeze)
    add5 = concatenate([fire5_expand1, fire5_expand2],axis=1)
    # mode='concat',
    # concat_axis=1)

    fire6_squeeze = Convolution2D(
        48,
        1,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire6_squeeze')(add5)
    fire6_expand1 = Convolution2D(
        192,
        1,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire6_expand1')(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192,
        3,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire6_expand2')(fire6_squeeze)
    add6 = concatenate([fire6_expand1, fire6_expand2],axis=1)
    # mode='concat',
    # concat_axis=1)

    fire7_squeeze = Convolution2D(
        48,
        1,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire7_squeeze')(add6)
    fire7_expand1 = Convolution2D(
        192,
        1,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire7_expand1')(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        192,
        3,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire7_expand2')(fire7_squeeze)
    add7 = concatenate([fire7_expand1, fire7_expand2],axis=1)
    # mode='concat',
    # concat_axis=1)

    fire8_squeeze = Convolution2D(
        64,
        1,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire8_squeeze')(add7)
    fire8_expand1 = Convolution2D(
        256,
        1,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire8_expand1')(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        256,
        3,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire8_expand2')(fire8_squeeze)
    add8 = concatenate([fire8_expand1, fire8_expand2],axis=1)
    # mode='concat',
    # concat_axis=1)

    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            name='maxpool8')(add8)

    fire9_squeeze = Convolution2D(
        64,
        1,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire9_squeeze')(maxpool8)
    fire9_expand1 = Convolution2D(
        256,
        1,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire9_expand1')(fire9_squeeze)
    fire9_expand2 = Convolution2D(
        256,
        3,
        activation='relu',
        # init='glorot_uniform',
        padding='same',
        name='fire9_expand2')(fire9_squeeze)

    add9 = concatenate([fire9_expand1, fire9_expand2],axis=1)
    # mode='concat',
    # concat_axis=1)

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(add9)
    conv10 = Convolution2D(
        nb_classes,
        1,
        # init='glorot_uniform',
        # border_mode='valid',
        padding='valid',
        name='conv10')(fire9_dropout)
    # The size should match the output of conv10
    avgpool10 = AveragePooling2D((13, 13), name='avgpool10')(conv10)

    flatten = Flatten(name='flatten')(avgpool10)
    softmax = Activation("softmax", name='softmax')(flatten)

    return Model(inputs=input_img, outputs=softmax)
    return Model()
    # return Model(input=input_img, output=conv10)


if __name__ == "__main__":
    sq = SqueezeNet(10, inputs=(375, 500, 3))
    sq.summary()
    plot_model(sq, to_file='model.png', show_shapes=True)