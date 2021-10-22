import copy
import pickle, gzip
import numpy as np

batch_size = 5000
input_layer_size = 784
hidden_layer_size = 100
output_layer_size = 10

with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

xl1 = [0 for i in range(input_layer_size)]
xl2 = [0 for i in range(hidden_layer_size)]
xl3 = [0 for i in range(output_layer_size)]
X = (xl1, xl2, xl3)

wl1 = [[np.random.randn() for i in range(input_layer_size)] for j in range(hidden_layer_size)]
wl2 = [[np.random.randn() for i in range(hidden_layer_size)] for j in range(output_layer_size)]
W = (wl1, wl2)

bl1 = [np.random.randn() for i in range(hidden_layer_size)]
bl2 = [np.random.randn() for i in range(output_layer_size)]
B = (bl1, bl2)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


xl1 = copy.deepcopy(train_set[0][0])
print("\n\n")
xl2 = sigmoid(np.dot(wl1, xl1) + bl1)
print(xl2)


