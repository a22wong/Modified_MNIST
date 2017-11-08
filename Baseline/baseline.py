# Alexander Wong
# 260602944
# COMP 551 Project 3
# Group:


import numpy as np
from sklearn.linear_model import LogisticRegression

train_x = "../Data/train_x.csv"
train_y = "../Data/train_x.csv"
train_x_short = "../Data/train_x_1000.csv"
train_y_short = "../Data/train_y_1000.csv"

def main():
    x = np.loadtxt(train_x_short, delimiter=",")  # load from text
    y = np.loadtxt(train_y_short, delimiter=",")
    # indices = [indx for indx,yi in enumerate(y) if yi > 0]
    # x = [x[indx] for indx in indices]
    # y = [y[indx] for indx in indices]


    logisticRegr = LogisticRegression(solver='lbfgs')
    logisticRegr.fit(x, y)
    predictions = logisticRegr.predict(x)

    print np.mean(predictions==y)


if __name__ == "__main__":
    main()