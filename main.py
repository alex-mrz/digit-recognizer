import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

data = pd.read_csv('/Users/Shared/Py_Projects/train.csv')

# print(data.head())    # check data

data = np.array(data)   # convert to numpy array
m, n = data.shape
np.random.shuffle(data)

data[:, 1:] = data[:, 1:].astype(np.float32) / 255.0   #normalize the data


data_dev = data[0:1000].T
Y_dev = data_dev[0]     #labels
X_dev = data_dev[1:n]       #features

data_train = data[1000:m].T
Y_train = data_train[0]     #labels
X_train = data_train[1:n]       #features

def init_params(hidden_size1=128, hidden_size2=64):
    W1 = np.random.randn(hidden_size1, 784) * np.sqrt(2  / 784)
    b1 = np.zeros((hidden_size1, 1))

    W2 = np.random.randn(hidden_size2, hidden_size1) * np.sqrt(2 / hidden_size1)
    b2 = np.zeros((hidden_size2, 1))

    W3 = np.random.randn(10, hidden_size2) * np.sqrt(2 / hidden_size2)
    b3 = np.zeros((10, 1))


    return W1, b1, W2, b2, W3, b3

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Prevent overflow
    return expZ / np.sum(expZ, axis=0, keepdims=True)  # Normalize column-wise

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return (Z > 0).astype(float)   #numpy changes boolean to 0 or 1

def forw_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)

    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # Y.size is number of training samples,  Y.max() + 1: num. of digits
    one_hot_Y[np.arange(Y.size), Y] = 1     #set the 3rd colomn to 1 if the label is 3
    one_hot_Y = one_hot_Y.T     #rows become classes(labels) and columns the samples (image)
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    m = Y.size
                                # Output layer gradients
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = W3.T.dot(dZ3)         # Backpropagation into second hidden layer
    dZ2 = dA2 * deriv_ReLU(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T.dot(dZ2)         # Backpropagation into first hidden layer
    dZ1 = dA1 * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3

    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def initialize_adam(params):
    # Initialize m and v for each parameter as zeros
    mo = {}
    v = {}
    keys = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
    for key, param in zip(keys, params):
        mo[key] = np.zeros_like(param)
        v[key] = np.zeros_like(param)
    return mo, v

def adam_update(params, grads, mo, v, t, alpha, beta1=0.9, beta2=0.999, epsilon=1e-8):
 
    keys = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
    updated_params = []
    for key, param, grad in zip(keys, params, grads):
        mo[key] = beta1 * mo[key] + (1 - beta1) * grad
        v[key] = beta2 * v[key] + (1 - beta2) * (grad ** 2)
        m_corrected = mo[key] / (1 - beta1 ** t)
        v_corrected = v[key] / (1 - beta2 ** t)
        param = param - alpha * m_corrected / (np.sqrt(v_corrected) + epsilon)
        updated_params.append(param)
    return tuple(updated_params), mo, v

def gradient_descent_adam(X, Y, iterations, alpha, start_fresh=True, checkpoint_file="params.npz", batch_size=128):
    
    if not start_fresh and os.path.exists(checkpoint_file):
        print("Resuming training from saved parameters.")
        checkpoint = np.load(checkpoint_file)
        W1, b1, W2, b2, W3, b3 = (checkpoint["W1"], checkpoint["b1"],
                                   checkpoint["W2"], checkpoint["b2"],
                                   checkpoint["W3"], checkpoint["b3"])
    else:
        print("Starting fresh training.")
        W1, b1, W2, b2, W3, b3 = init_params()
    
    params = (W1, b1, W2, b2, W3, b3)
    m = X.shape[1]  
    m_adam, v_adam = initialize_adam(params)
    t = 0  # time step for bias correction
    
    for i in range(1, iterations + 1):
        # Create a mini-batch by randomly selecting indices
        batch_indices = np.random.choice(m, batch_size, replace=False)
        X_batch = X[:, batch_indices]
        Y_batch = Y[batch_indices]
        
        # Forward propagation on mini-batch
        Z1, A1, Z2, A2, Z3, A3 = forw_prop(params[0], params[1], params[2], params[3], params[4], params[5], X_batch)
        # Compute gradients on mini-batch
        grads = back_prop(Z1, A1, Z2, A2, Z3, A3, params[2], params[4], X_batch, Y_batch)
        t += 1  
        
        # Update parameters using Adam
        params, m_adam, v_adam = adam_update(params, grads, m_adam, v_adam, t, alpha)
        
        # Every 100 iterations, evaluate on the full training set and save checkpoint
        if i % 100 == 0:
            Z1_full, A1_full, Z2_full, A2_full, Z3_full, A3_full = forw_prop(params[0], params[1], params[2], params[3], params[4], params[5], X)
            predictions = get_predictions(A3_full)
            acc = get_accuracy(predictions, Y)
            print('Iteration:', i, "Training Accuracy:", acc)
            np.savez(checkpoint_file, W1=params[0], b1=params[1], W2=params[2], b2=params[3], W3=params[4], b3=params[5])
    
    return params


W1, b1, W2, b2, W3, b3 = gradient_descent_adam(X_train, Y_train, 1000, 0.0005, start_fresh=False, batch_size=128)










