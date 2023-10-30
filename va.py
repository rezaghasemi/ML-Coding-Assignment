#!/usr/bin/env python3

import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def predict(X, w, y=None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    # TODO: Your code here
    y_hat = np.dot(X,w)
    loss = np.dot((y - y_hat).T,(y - y_hat))
    M_val = y_hat.shape[0]
    risk = 1/M_val*np.sum(np.abs(y_hat - y))

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
            Gradient_batch = 1/batch_size*np.dot(X_batch.T, y_hat_batch - y_batch)
            w += - alpha*Gradient_batch
        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        # 2. Perform validation on the validation set by the risk
        # 3. Keep track of the best validation epoch, risk, and the weights
        training_loss = 1/(2*N_train)*loss_this_epoch
        _ , _ , risk_val = predict(X_val, w, y_val)
        losses_train = np.append(losses_train, training_loss)
        risks_val = np.append(risks_val, risk_val)
        if risk_val < risk_best:
          risk_best = risk_val
          epoch_best = epoch
          w_best = w
    # Return some variables as needed
    return epoch_best, risk_best, w_best, losses_train, risks_val


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
epoch_best, risk_best, w_best, losses_train, risks_val = train(X_train, y_train, X_val, y_val)
# Perform test by the weights yielding the best validation performance
_ , _ , risk_test = predict(X_test, w_best, y_test)
# Report numbers and draw plots as required.
print('Epoch_best=', epoch_best, '\nRisk_best_validation = ', risk_best, '\nRisk_test= ', risk_test)

iterations = list(range(1, MaxIter + 1))
plt.plot(iterations, losses_train, color="blue", label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Training_Loss_#2_a' + '.jpg')

plt.plot(iterations, risks_val, color="red", label='Validation Risk')
plt.title('Validation Risk Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation Risk')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Validation_Risk_#2_a' + '.jpg')