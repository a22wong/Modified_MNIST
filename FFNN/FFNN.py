# Alexander Wong
# 260602944
# COMP 551 Project 3
# Group: Not Very Accurate
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

from random import seed
import random
from math import exp
import numpy as np
import copy
from sklearn.preprocessing import normalize


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random.uniform(0,1) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random.uniform(0,1) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def reMapY(y):
    # re-map y to 0-40 classes instead of kaggle classes
    kaggle_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32,
                      35, 36, 40, 42,
                      45, 48, 49, 54, 56, 63, 64, 72, 81]
    output = np.array(y)
    for i, j in enumerate(y):
        output[i] = kaggle_classes.index(j)
    return output


def main():
    # load data
    print "Loading data..."
    train_x = "../Data/train_x_head1k.csv"
    train_y = "../Data/train_y_head1k.csv"
    test_x = "../Data/train_x_tail1k.csv"
    test_y = "../Data/train_y_tail1k.csv"
    x_train = np.loadtxt(train_x, delimiter=",")
    y_train = np.loadtxt(train_y, delimiter=",")
    x_test = np.loadtxt(test_x, delimiter=",")
    y_test = np.loadtxt(test_y, delimiter=",")

    # re format y to fit FFNN algorithm
    print "Preprocessing..."
    y_train = reMapY(y_train)
    y_train = np.matrix(y_train)
    y_test = reMapY(y_test)
    expected = copy.deepcopy(y_test)
    y_test = np.matrix(y_test)


    print "Normalizing..."
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # add y as last column of dataset
    dataset = np.concatenate((x_train, y_train.T), 1)
    dataset = dataset.tolist()

    print "Initializing network..."
    # initialize network
    n_inputs = len(dataset[0]) - 1
    n_outputs = 40  # len(set([row[-1] for row in dataset]))
    network = initialize_network(n_inputs, 10, n_outputs)

    print "Training..."
    # train
    # try momentum learning rate
    # try relu activation: max{0,z}
    train_network(network, dataset, 1, 10, n_outputs)
    # for layer in network:
    #     print(layer)

    # predict
    print "Predicting..."
    predictions = []
    for row in dataset:
        prediction = predict(network, row)
        predictions.append(prediction)
        print('Expected=%d, Predicted=%d' % (row[-1], prediction))

    print "Printing predictions..."
    np.savetxt("FFNN.csv", predictions, fmt='%d', delimiter=",")

    # measure accuracy
    print "Accuracy: "+str(accuracy_metric(expected, predictions))+"%"


if __name__ == "__main__":
    main()
