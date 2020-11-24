import matplotlib.pyplot as plt


def show_history(history,model_name='cnn',):  # 绘制mse图像

    plt.figure(1)
    plt.plot(history.history['accuracy'],'-r')
    plt.plot(history.history['val_accuracy'],':b')
    # plt.plot(history0.history['acc']+history1.history['acc'],'-r')
    # plt.plot(history0.history['val_acc']+history1.history['val_acc'],':b')
    plt.title(model_name+' accuracy')
    plt.ylabel('acc')
    plt.xlabel('Epoch')
    # plt.legend(['Train_loss', 'Test_loss','Train_acc', 'Test_acc'], loc='upper left')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy_'+model_name+'.jpg',dpi=200)

    plt.figure(2)
    plt.plot(history.history['loss'],'-r')
    plt.plot(history.history['val_loss'],':b')
    plt.title(model_name+' loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss_'+model_name+'.jpg',dpi=200)
    # plt.show()

    print(history.history['accuracy'])
    print(history.history['val_accuracy'])
    print(history.history['loss'])
    print(history.history['val_loss'])
