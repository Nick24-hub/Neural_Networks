import gzip
import pickle
import numpy as np

epochs = 30
batch_size = 10
input_layer_size = 784
hidden_layer_size = 100
output_layer_size = 10
learning_rate = 0.001

with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

x_in = np.asarray(train_set[0])
x_in = x_in.reshape((len(x_in), input_layer_size, 1))
x_label = np.asarray(train_set[1])

p = np.random.permutation(len(x_in))
x_in = x_in[p]
x_label = x_label[p]

test_in = np.asarray(test_set[0])
test_in = test_in.reshape((len(test_in), input_layer_size, 1))
test_label = np.asarray(test_set[1])

xl1 = np.zeros((input_layer_size, 1))
xl2 = np.zeros((hidden_layer_size, 1))
xl3 = np.zeros((output_layer_size, 1))
X = [xl1, xl2, xl3]

wl1 = np.random.normal(0, 1 / np.sqrt(input_layer_size), (hidden_layer_size, input_layer_size))
wl2 = np.random.normal(0, 1 / np.sqrt(hidden_layer_size), (output_layer_size, hidden_layer_size))
W = [wl1, wl2]

bl1 = np.random.normal(0, 1 / np.sqrt(input_layer_size), (hidden_layer_size, 1))
bl2 = np.random.normal(0, 1 / np.sqrt(input_layer_size), (output_layer_size, 1))
B = [bl1, bl2]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


for epoch in range(epochs):
    print("Epoch", epoch + 1)
    for batch in range(len(x_in) // batch_size):
        deltal1 = np.zeros((hidden_layer_size, input_layer_size))
        deltal2 = np.zeros((output_layer_size, hidden_layer_size))
        Delta = (wl1, wl2)

        betal1 = np.zeros((hidden_layer_size, 1))
        betal2 = np.zeros((output_layer_size, 1))
        Beta = (bl1, bl2)
        for index in range(batch * batch_size, (batch + 1) * batch_size):
            xl1 = x_in[index]
            z = np.dot(wl1, xl1) + bl1
            y = sigmoid(z)
            xl2 = np.copy(y)
            z = np.dot(wl2, xl2) + bl2
            y = softmax(z)
            xl3 = np.copy(y)
            t = np.zeros((output_layer_size, 1))
            t[x_label[index]] = 1
            error_l3 = xl3 - t
            deltal2 += np.dot(error_l3, xl2.T)
            betal2 += error_l3
            error_l2 = np.dot(np.dot(xl2, (1 - xl2).T), np.dot(wl2.T, error_l3))
            deltal1 += np.dot(error_l2, xl1.T)
            betal1 += error_l2
        wl1 -= deltal1 * learning_rate / batch_size
        bl1 -= betal1 * learning_rate / batch_size
        wl2 -= deltal2 * learning_rate / batch_size
        bl2 -= betal2 * learning_rate / batch_size
    acuracy = 0
    for example in range(len(test_in)):
        z = np.dot(wl1, test_in[example]) + bl1
        y = sigmoid(z)
        z = np.dot(wl2, y) + bl2
        y = softmax(z)
        result = np.argmax(y)
        if result == test_label[example]:
            acuracy = acuracy + 1
    rate = acuracy * 100 / len(test_in)
    print("Rate:", rate)
