import csv
import numpy as np
import matplotlib.pyplot as plt

train_x = "../Data/train_x.csv"
train_y = "../Data/train_x.csv"
train_x_short = "../Data/train_x_short.csv"
train_y_short = "../Data/train_y_short.csv"

def reprocess():
    with open(train_x, 'r+') as readfile:
        with open(train_x_short, 'w') as writefile:
            reader = csv.reader(readfile)
            writer = csv.writer(writefile)
            count = 0
            for row in reader:
                writer.writerow(row)
                count += 1
                if count == 10:
                    break

    with open(train_y, 'r+') as readfile:
        with open(train_y_short, 'w') as writefile:
            reader = csv.reader(readfile)
            writer = csv.writer(writefile)
            count = 0
            for row in reader:
                writer.writerow(row)
                count += 1
                if count == 10:
                    break

# reprocess()
x = np.loadtxt("../Data/train_x_short.csv", delimiter=",") # load from text
y = np.loadtxt("../Data/train_y.csv", delimiter=",")
print "Raw:"
print x[0]
x = x.reshape(-1, 64, 64) # reshape
y = y.reshape(-1, 1)
print "Reshaped"
print x[0]
plt.imshow(x[5]) # to visualize only
plt.show()
