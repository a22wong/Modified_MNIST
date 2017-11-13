# Alexander Wong
# 260602944
# COMP 551 Project 3
# Group:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

def reMapY(y):
    # re-map y to 0-40 classes instead of kaggle classes
    kaggle_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
    output = np.array(y)
    for i, j in enumerate(y):
        output[i] = kaggle_classes.index(j)
    return output

def main():
    print "Loading training data..."
    train_x = "../Data/train_x_train_small.csv"
    train_y = "../Data/train_y_train_small.csv"
    x_train = np.loadtxt(train_x, delimiter=",")
    y_train = np.loadtxt(train_y, delimiter=",")

    print "Loading validation data..."
    validate_x = "../Data/train_x_validate_small.csv"
    validate_y = "../Data/train_y_validate_small.csv"
    x_validate = np.loadtxt(validate_x, delimiter=",")
    y_validate = np.loadtxt(validate_y, delimiter=",")

    print "Loading testing data..."
    test_x = "../Data/train_x_test_small.csv"
    test_y = "../Data/train_y_test_small.csv"
    x_test = np.loadtxt(test_x, delimiter=",")
    y_test = np.loadtxt(test_y, delimiter=",")

    print "Normalizing input..."
    x_train = normalize(x_train)
    x_validate = normalize(x_validate)
    x_test = normalize(x_test)

    print "Preprocessing output..."
    y_train = reMapY(y_train)
    y_validate = reMapY(y_validate)
    y_test = reMapY(y_test)
    y_train = np.matrix(y_train)
    y_validate = np.matrix(y_validate)
    y_test = np.matrix(y_test)

    # trying to exlclude certain classes
    # indices = [indx for indx,yi in enumerate(y) if yi > 0]
    # x = [x[indx] for indx in indices]
    # y = [y[indx] for indx in indices]



    print "Fitting..."
    logisticRegr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    logisticRegr.fit(x_train, y_train)

    print "Predicting..."
    predictions = logisticRegr.predict(x_test)

    # print "Printing..."
    # np.savetxt("baseline.csv", predictions, fmt='%d', delimiter=",")

    for i,p in enumerate(predictions):
        print'Expected: %d, Predicted: %d' % (y_test[i], p)

    print "Score: " + str(logisticRegr.score(x_test, y_test))
    print "Accuracy: " + str(accuracy_score(y_test, predictions))


if __name__ == "__main__":
    main()