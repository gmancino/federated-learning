# ------------------------
# Distributed Logistic Regression with Adversaries Test
# ------------------------

# Many things were taken from https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16


# Import necessary packages
import os
import time
import math
import argparse
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from MulticlassLogReg import MulticlassLogReg
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Define custom functions
def create_agents(x, y, k, percent_test):
    '''
    Function for creating 'agents' with separate data on each agent

    INPUTS:
    x = matrix where each row is a data point (n x d - matrix)
    y = vector with corresponding class label (d - vector)
    k = number of agents
    percent_test = percentage of data points to be left out for testing

    OUTPUTS:
    x_train_agents = tensor where each first index hosts an agents training data (k x (1-percent_test)*(n/k) x d - matrix)
    y_train_agents = corresponding class for each agent ( (1-percent_test)*(n/k) x k vector )
    '''

    [n, d] = x.shape

    # Save percent_test% of data points for testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percent_test, random_state=10)

    # Get shape for splitting data
    m = x_train.shape[0]

    # Cut up data for 'agents' to use
    k = k
    print('[INFO] Distributing data to ' + str(k) + ' \'agents\'...')

    # Allocate space
    num = math.floor(m/k)
    x_train_agents = np.zeros((k, num, d))
    y_train_agents = np.zeros((num, k))

    # Chop up data into different clusters
    for i in range(k):
        if i == 0:
            x_train_agents[i, :, :] = x_train[0:num, :]
            y_train_agents[:, i] = y_train[0:num]
        elif (i < k - 1):
            x_train_agents[i, :, :] = x_train[(i*num):((i+1)*num), :]
            y_train_agents[:, i] = y_train[(i*num):((i+1)*num)]
        else:
            x_train_agents[i, :, :] = x_train[(i*num):m, :]
            y_train_agents[:, i] = y_train[(i*num):m]

    print('[INFO] Done sorting...')

    return[x_train_agents, y_train_agents, x_test, y_test]


def view_data(x):
    plt.imshow(x.reshape(28, 28))
    return None


def one_hot_encoding(y, num_classes):
    '''Encode vector for multiclass regression into matrix'''

    ln = len(y)

    y = y.astype(int)

    Y = np.zeros((ln, num_classes))

    for i in range(ln):
        Y[i, y[i]] = 1

    return Y


def softmax(z):

    # Subtract max for numerical stability
    z -= np.max(z)

    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
    return sm

def getLoss(w,x,y,lam):
    m = x.shape[0] #First we get the number of training examples
    y_mat = one_hot_encoding(y, 10) #Next we convert the integer class coding into a one-hot representation
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w #And compute the gradient for that loss
    return loss,grad


def predict(x, w, b):
    raw = np.dot(x, w) + b
    probability = softmax(raw)
    prediction = np.argmax(probability, axis=1)
    return [prediction, probability]


def multiclass_log_reg(x, y, k, lr, lam, iterations):

    [n, d] = x.shape

    w = np.zeros(shape=(d, k))
    b = np.zeros(shape=(n, k))

    losses = []
    for i in range(0, iterations):
        loss, grad = getLoss(w, x, y, lam)
        losses.append(loss)

        # Simple step size
        w = w - (lr * grad)

    return [w, losses]


# Load data for problem
print('[INFO] Loading data...')
x, y = load_svmlight_file("mnist")
x = x.todense()

# Normalize data
x = np.array(x) / np.max(x)

# Fix our data to make square bois
concat = np.zeros((x.shape[0], 4))
x = np.append(x, concat, axis=1)

print('[INFO] Data has been loaded. Splitting into training and testing sets...')

# Split data into agents and testing data
[x_train_agents, y_train_agents, x_test, y_test] = create_agents(x, y, 10, 0.15)

test = MulticlassLogReg(x_train_agents[0], y_train_agents[:, 0], np.zeros((x_train_agents[0].shape[1], 10)), np.zeros((10)), T=1000, tol=0.5, alpha=0.25, lam1=1, lam2=1, num_classes=10)
test.optimization()

[pred, prob] = predict(x_train_agents[1], test.w, test.b)

print('Accuracy: ', sum(pred == y_train_agents[:, 1])/len(pred))
