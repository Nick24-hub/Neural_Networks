import gzip
import pickle
import numpy as np

epochs = 30
batch_size = 10
input_layer_size = 784
output_layer_size = 10
learning_rate = 0.001

with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

x_in = np.asarray(np.concatenate((train_set[0], valid_set[0])))
x_in = x_in.reshape((len(x_in), input_layer_size, 1))
x_label = np.asarray(np.concatenate((train_set[1], valid_set[1])))

p = np.random.permutation(len(x_in))
x_in = x_in[p]
x_label = x_label[p]

test_in = np.asarray(test_set[0])
test_in = test_in.reshape((len(test_in), input_layer_size, 1))
test_label = np.asarray(test_set[1])

W = np.random.randn(output_layer_size, input_layer_size)
bias = np.random.randn(output_layer_size, 1)

for epoch in range(epochs):
    print("Epoch", epoch + 1)
    for batch in range(len(x_in) // batch_size):
        delta = np.zeros((output_layer_size, input_layer_size))
        beta = np.zeros((output_layer_size, 1))
        for index in range(batch * batch_size, (batch + 1) * batch_size):
            z = np.dot(W, x_in[index]) + bias
            t = []
            for digit in range(10):
                if x_label[index] == digit:
                    t.append(1)
                else:
                    t.append(-1)
            t = np.array(t)
            t = t.reshape(10, 1)
            delta += np.dot(t - z, np.transpose(x_in[index])) * learning_rate
            beta += (t - z) * learning_rate
        W += delta
        bias += beta
    rate = 0
    for example in range(len(test_in)):
        z = np.dot(W, test_in[example]) + bias
        result = np.argmax(z)
        if result == test_label[example]:
            rate = rate + 1
    rate = rate * 100 / len(test_in)
    print("Rate:", rate)
