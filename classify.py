import numpy as np

from get_data import display_img
from get_data import get_data

data = [get_data(n) for n in range(10)]

# 70% training, 30% testing.
training = [d[:700] for d in data]
test = [d[700:] for d in data]


class Classifier(object):
    def __init__(self, training, test):
        self.training = training
        self.test = test
        self.train()

    def predict(self, img):
        """
        forward pass through the network
        :param img: 28*28 dimensional np array representing an image
        :return: 0-9
        """
        vects = [  # yes i know this is I_10
            np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.asarray([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.asarray([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            np.asarray([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ]
        prediction = self.forward_pass(img)
        dists = [np.linalg.norm(vect - prediction) for vect in vects]
        return np.argmin(dists)

    def forward_pass(self, img):
        """
        some matrix mult or smthng.
        :param img: 28*28 dimensional np array representing an image
        :return: vector - [1 0 ... 0 0] through [0 0 ... 0 1]
        """
        pass

    def train(self):
        pass

    def accuracy(self):
        n_correct = 0.0
        total = 0.0
        for label in range(10):
            total += len(self.test[label])
            for img in self.test[label]:
                if self.predict(img) == label:
                    n_correct += 1
        return n_correct / total


c = Classifier(training, test)
