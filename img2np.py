import numpy as np
import os
import cv2
from multiprocessing import Queue, Process, Pool


def img2num(oneImage):  #按文件名返回图片numpy矩阵
    x = cv2.imread(oneImage)
    # print(oneImage)
    return x


if __name__ == "__main__":
    name='cherry'
    path = 'D:\\ftpserver\\attention\\data\\'
    # path_list = os.listdir(path)
    # print(path_list)
    KIND = ['Cherry___healthy', 'Cherry___Powdery_mildew']

    X=[]
    Y=[]

    label = 0
    for i in KIND:
        path_kind = path + i
        filenames = os.listdir(path_kind)
        pathnames = [ os.path.join(path_kind, filename) for filename in filenames]

        pool = Pool()
        x = pool.map(img2num, pathnames)
        x = np.array(x)

        one_np = np.ones(len(pathnames))
        y=one_np*label

        print(x.shape)
        print(y.shape)

        X.append(x)
        Y.append(y)

        label=label+1


    X_np=np.concatenate(X,axis=0)
    Y_np=np.concatenate(Y,axis=0)
    print(X_np.shape)
    print(Y_np.shape)
    np.save('X_'+name, X_np)
    np.save('Y_'+name, Y_np)
