#!/usr/bin/env python3

import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt

def predict(X, w, y=None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    # TODO: Your code here
    y_hat = X @ w
    loss = np.linalg.norm(y_hat-y)**2
    risk = np.linalg.norm(y_hat-y, ord=1)/y_hat.shape[0]

    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):

        loss_this_epoch = 0
        
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            w = w - alpha * 1/batch_size * X_batch.T @ (y_hat_batch - y_batch)
        
        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        losses_train.append(loss_this_epoch/(2*N_train))

        # 2. Perform validation on the validation set by the risk
        risk = predict(X_val,w,y_val)[-1]
        risks_val.append(risk)


        # 3. Keep track of the best validation epoch, risk, and the weights
        if risk < risk_best:
            risk_best = risk
            epoch_best = epoch
            w_best = w
    # Return some variables as needed
    return w_best, epoch_best, risk_best, losses_train, risks_val


############################
# Main code starts here
############################
# Load data
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

# print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val = X_[300:400]
y_val = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# setting

alpha = 0.001      # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay


# TODO: Your code here
w_best, epoch_best, risk_best, losses_train, risks_val = train(X_train, y_train, X_val, y_val)
risk_test = predict(X_test, w_best, y_test)[-1]
# Perform test by the weights yielding the best validation performance
# w_best, epoch_best, risk_best, losses_train
# print(f"the best weight i {results[0]} and happens at eppoch number {results[1]} and has {results[2]} risk")

print(f"Epoch at which we get the maximum: {epoch_best}")

print(f"validation performance in that epoch is {risk_best}")

print(f"test performance (risk) in that epoch {risk_test}")

# Report numbers and draw plots as required.

# w_best, epoch_best, risk_best, losses_train, risks_val
plt.plot(range(MaxIter), losses_train, 'r--')
plt.ylabel('Train Loss')
plt.xlabel('Epoch')
# plt.plot(range(len(results[3])), results[3], 'r--')
plt.show()

plt.plot(range(MaxIter), risks_val, 'b--')

plt.ylabel('Risk')
plt.xlabel('Epoch')

plt.show()