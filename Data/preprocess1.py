from subprocess import call
import numpy as np
import matplotlib.pyplot as plt

train_x = "../Data/train_x.csv"
train_y = "../Data/train_x.csv"
train_x_short = "../Data/train_x_short.csv"
train_y_short = "../Data/train_y_short.csv"

def callHead(n, outfile_x, outfile_y):
    n = "-"+str(n)
    call(["head", n, train_x, ">", outfile_x])
    call(["head", n, train_y, ">", outfile_y])

def callTail(n, outfile_x, outfile_y):
    n = "-"+str(n)
    call(["tail", n, train_x, ">", outfile_x])
    call(["tail", n, train_y, ">", outfile_y])

def callRandom(n):
    return

def main():
    # callHead(200, train_x_short, train_y_short)
    x = np.loadtxt("../Data/train_x_short.csv", delimiter=",") # load from text
    y = np.loadtxt("../Data/train_y_short.csv", delimiter=",")
    x = x.reshape(-1, 64, 64) # reshape
    y = y.reshape(-1, 1)

    plot_n = 2
    for index, (image, label) in enumerate(zip(x[0:plot_n], y[0:plot_n])):
        plt.subplot(1,plot_n,index+1)
        plt.title("Training: %i\n" % label)
        plt.imshow(x[index])
    plt.show()

if __name__ == "__main__":
    main()