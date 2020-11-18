from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPool2D,GlobalAveragePooling2D,Dropout
from tensorflow.keras.layers import concatenate,Dense
# from tensorflow.keras.utils import plot_model

def fire_model(x, s_1x1, e_1x1, e_3x3, fire_name):
    # squeeze part
    squeeze_x = Conv2D(kernel_size=(1,1),filters=s_1x1,padding='same',activation='relu',name=fire_name+'_s1')(x)
    # expand part
    expand_x_1 = Conv2D(kernel_size=(1,1),filters=e_1x1,padding='same',activation='relu',name=fire_name+'_e1')(squeeze_x)
    expand_x_3 = Conv2D(kernel_size=(3,3),filters=e_3x3,padding='same',activation='relu',name=fire_name+'_e3')(squeeze_x)
    # expand = merge([expand_x_1, expand_x_3], mode='concat', concat_axis=3)
    expand = concatenate([expand_x_1, expand_x_3],axis=3)

    return expand

def SqueezeNet(nb_classes,inputs=(224, 224,3)):
    input_img = Input(shape=inputs)
    conv1 = Conv2D(strides = 2, filters=96, kernel_size=(7,7), padding='same', activation='relu',data_format='channels_last')(input_img)
    poo1 = MaxPool2D((2,2))(conv1)
    fire2 = fire_model(poo1, 16, 64, 64,'fire2')
    fire3 = fire_model(fire2, 16, 64, 64,'fire3')
    fire4 = fire_model(fire3, 32, 128, 128,'fire4')
    pool2 = MaxPool2D((2,2))(fire4)
    fire5 = fire_model(pool2, 32, 128, 128,'fire5')
    fire6 = fire_model(fire5, 48, 192, 192,'fire6')
    fire7 = fire_model(fire6, 48, 192, 192,'fire7')
    fire8 = fire_model(fire7, 64, 256, 256,'fire8')
    pool3 = MaxPool2D((2,2))(fire8)
    fire9 = fire_model(pool3, 64, 256, 256,'fire9')
    dropout1 = Dropout(0.5)(fire9)
    conv10 = Conv2D(kernel_size=(1,1), filters=100, padding='same', activation='relu')(dropout1)
    gap = GlobalAveragePooling2D()(conv10)
    out=Dense(nb_classes,activation='softmax')(gap)
    
    return Model(inputs=input_img, outputs=out)

if __name__ == "__main__":
    sq = SqueezeNet(10, inputs=(375, 500, 2))
    sq.summary()
    # plot_model(sq, to_file='model_1d.png', show_shapes=True)