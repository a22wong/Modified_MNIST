# -*- coding: utf-8 -*-
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from scipy.ndimage import zoom
from scipy.misc import imresize

x = np.loadtxt("train_x_100.csv", delimiter=",") # load from text
y = np.loadtxt("train_y_100.csv", delimiter=",")

def Normalize(x):
    #normal_x = X_normalized = preprocessing.normalize(x, norm='1')
    normal_x = Normalizer().fit_transform(x)
    return normal_x

def Scale(x, scaleSize):
    small_x = zoom(x, scaleSize)
    #small_x = imresize(x, scaleSize, interp='nearest', mode=None)
    return small_x


    

def main():
    x = np.loadtxt("train_x_100.csv", delimiter=",") # load from text
    y = np.loadtxt("train_y_100.csv", delimiter=",")
    normal_x = Normalize(x)
    small_x = Scale(normal_x, 0.5)
    # callHead(200, train_x_short, train_y_short)
    #np.loadtxt("train_x_100.csv", delimiter=",") # load from text
    x = small_x.reshape(-1, 128, 64) # reshape
    #y = y.reshape(-1, 1)

    plot_n = 1
    for index, (image, label) in enumerate(zip(x[0:plot_n], y[0:plot_n])):
        plt.subplot(1,plot_n,index+1)
        plt.title("Training: %i\n" % label)
        plt.imshow(x[index])
    plt.show()

if __name__ == "__main__":
    main()
    

#train_x = "train_x_1000.csv" #../Data/
#train_y = "train_x_1000.csv" #../Data/
#train_x_short = "train_x_100.csv" #../Data/
#train_y_short = "train_y_100.csv" #../Data/
#
#def callHead(n, outfile_x, outfile_y):
#    n = "-"+str(n)
#    call(["head", n, train_x, ">", outfile_x])
#    call(["head", n, train_y, ">", outfile_y])
#
#def callTail(n, outfile_x, outfile_y):
#    n = "-"+str(n)
#    call(["tail", n, train_x, ">", outfile_x])
#    call(["tail", n, train_y, ">", outfile_y])
#
#def callRandom(n):
#    return