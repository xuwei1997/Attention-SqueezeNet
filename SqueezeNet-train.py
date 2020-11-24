# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

from SqueezeNet3 import SqueezeNet
import numpy as np
from tensorflow.keras.optimizers import Adam ,SGD
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.datasets import cifar10
from showHistory import show_history

def prepare_data():  # 取数据
    print("prepare data")

    X=np.load('X_cherry.npy')
    Y=np.load('Y_cherry.npy')

    print(X.shape)
    print(Y.shape)

    # X=np.transpose(X, (0,3,1,2))


    X=X/255
    Y=to_categorical(Y)
    

    X,Y = shuffle(X,Y) 

    X_train=X[:1500]
    Y_train=Y[:1500]
    X_test=X[1500:]
    Y_test=Y[1500:]
    return(X_train,Y_train,X_test,Y_test)


def prepare_data_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # print(x_train.shape,y_train.shape)
    x_train=x_train/32
    x_test=x_test/32
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    print(y_train.shape,y_test.shape)
    return(x_train,y_train,x_test,y_test)


if __name__ == "__main__":
    x_train,y_train,x_test,y_test  = prepare_data()
    model=SqueezeNet(nb_classes=2,inputs=(256, 256,3))

    # x_train,y_train,x_test,y_test  = prepare_data_cifar10()
    # model=SqueezeNet(nb_classes=10,inputs=(32, 32, 3))

    # model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)
    print("compile")
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=SGD(),metrics=['accuracy'])


    print("fit")
    Hist = model.fit(x_train, y_train, epochs=100, batch_size=64,validation_data=(x_test, y_test),verbose=1)

    # model.save_weights('weight.h5')
    # print(Hist.history)
    show_history(Hist,model_name='c10,lr=0.001,ep=100,b_s=128')
