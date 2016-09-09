import gzip
import matplotlib.pyplot as plt
import numpy as np
import pickle
import theano
import theano.tensor as T


def get_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f)
        except:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set


def display_img(arr):
    arr = arr.reshape((28, 28))
    plt.imshow(arr, cmap='Greys_r')
    plt.show()


def get_xy(imgs, labels):
    x = theano.shared(
        value=np.asarray(imgs, dtype=theano.config.floatX),
        borrow=True,
    )
    y = T.cast(theano.shared(
        value=np.asarray(labels, dtype=theano.config.floatX),
        borrow=True,
    ), 'int32')
    return x, y
