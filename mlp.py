import numpy as np
import theano
import theano.tensor as T
from classifier import Classifier


class HiddenLayer(object):
    def __init__(self, n_in, n_out, rng, activation=T.tanh):
        pass


class MLP(Classifier):
    def __init__(self, n_in, n_hidden, n_out, rng):
        self.hidden = HiddenLayer(
            n_in,
            n_hidden,
            rng,
        )
