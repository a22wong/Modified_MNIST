# -*- coding: utf-8 -*-
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from scipy.ndimage import zoom
from scipy.misc import imresize
import scipy.misc # to visualize only
import resize_data

def Normalize(x):
    #normal_x = preprocessing.normalize(x, norm='l2')
    #normal_x = Normalizer().fit_transform(x)
    m,n = x.shape
    normal_x = np.empty([m,n])
    for i in range(0, m):
        for j in range(0, n):
            if x[i,j] >= 240:
                normal_x[i,j] = 1
            else:
                normal_x[i,j] = 0
        
                 
    return normal_x



def Scale(x, scaleSize):
    small_x = zoom(x, scaleSize)
    #small_x = imresize(x, scaleSize, interp='nearest', mode=None)
    return small_x
    



def main():    
    # load train data
    print "Loading data..."
    x = "train_x.csv"
    y = "train_y.csv"
    KagX = "test_x.csv"
    train_data = np.loadtxt(x, delimiter=",").astype(np.float32)
    train_labels = np.loadtxt(y, dtype=int, delimiter=",")
    Kag_data = np.loadtxt(KagX, delimiter=",").astype(np.float32)
    
    print "Normalizing data..."
    X_train_norm = Normalize(train_data)
    Kag_norm = Normalize(Kag_data)
    
    print "Splitting data..."
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_norm, train_labels, test_size=0.30)
    
    print "Printing X data..."
    np.savetxt("X_train.csv", X_train, fmt='%f', delimiter=',')
    np.savetxt("X_test.csv", X_test, fmt='%f', delimiter=',')
    np.savetxt("Kag_test.csv", Kag_norm, fmt='%f', delimiter=',')
    
    print "Printing Y data..."
    np.savetxt("Y_train.csv", Y_train, fmt='%f', delimiter=',')
    np.savetxt("Y_test.csv", Y_test, fmt='%f', delimiter=',')
    
    
    x = np.loadtxt("x_train_1000.csv", delimiter=",")#.astype(np.float32)
    y = np.loadtxt("y_Train_1000.csv", delimiter=",")
    x = Normalize(x)
    # callHead(200, train_x_short, train_y_short)
    #np.loadtxt("train_x_100.csv", delimiter=",") # load from text
    x = resize_data.resize_images(x, 28)
    x = x.reshape(-1, 28, 28) # reshape
    y = y.reshape(-1, 1)
    
    plot_n = 2
    for index, (image, label) in enumerate(zip(x[0:plot_n], y[0:plot_n])):
        plt.subplot(1,plot_n,index+1)
        plt.title("Training: %i\n" % label)
        plt.imshow(x[index])
    plt.show()

        
#    for i in range(10000):
#        plt.imshow(x[i])
#        plt.show()
#        import time
#        time.sleep(3)

if __name__ == "__main__":
    main()
    
