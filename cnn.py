import numpy as np
import theano
import theano.tensor as T
from classifier import Classifier
from log_reg import LogisticRegression
from mlp import HiddenLayer
from mlp import MLP
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool


class LeNetConvPoolLayer(object):
    def __init__(self, rng, filter_shape, image_shape, poolsize=(2,2)):
        """
        :param rng: \o_O/
        :param filter_shape: (n_filters, n_input_feature_maps, filter height, filter width)
        :param image_shape: (batch_size, n_input_feature_maps, img height, img width
        :param poolsize:
        """
        assert filter_shape[1] == image_shape[1]
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize)
        W_bound = np.sqrt(6.0 / (fan_in + fan_out))
        self.W = theano.shared(
            value=np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX,
            ),
            borrow=True,
        )
        b_values = np.zeros(shape=(filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize=poolsize
        self.params = [self.W, self.b]

    def output(self, input):
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=self.filter_shape,
            input_shape=self.image_shape
        )
        pool_out = pool.pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True,
        )
        lin_output = pool_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return T.tanh(lin_output)


class ConvolutionalNeuralNetwork(Classifier):
    def __init__(self, rng, batch_size, nkerns=(20,50)):
        self.batch_size = batch_size
        # 28x28 -> (24x24) // 2 = 12x12
        self.layer0 = LeNetConvPoolLayer(
            rng=rng,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
        )
        # 12x12 -> (8x8) // 2 = 4x4
        self.layer1 = LeNetConvPoolLayer(
            rng=rng,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5)
        )
        # TODO: make this an MLP rather than a hidden layer -> LogReg
        # self.layer2 = MLP()
        self.layer2 = HiddenLayer(
            rng=rng,
            n_in=nkerns[1]*4*4,
            n_out=500,
            activation=T.tanh,
        )
        self.layer3 = LogisticRegression(
            n_in=500,
            n_out=10,
        )

    def pre_logreg_output(self, x):
        layer0_input = x.reshape((self.batch_size, 1, 28, 28))
        l0_output = self.layer0.output(layer0_input)
        l1_output = self.layer1.output(l0_output)

        l2_input = l1_output.flatten(2)
        l2_output = self.layer2.output(l2_input)
        return l2_output

    def negative_log_likelihood(self, x, y):
        output = self.pre_logreg_output(x)
        return self.layer3.negative_log_likelihood(output, y)

    def pred_label(self, x):
        output = self.pre_logreg_output(x)
        return self.layer3.pred_label(output)

    def errors(self, x, y):
        output = self.pre_logreg_output(x)
        return self.layer3.errors(output, y)

    def train(self, train_x, train_y, test_x, test_y, valid_x, valid_y, alpha=0.13, batch_size=500, l1_reg=0., l2_reg=0.0, n_epochs=1000):
        x = T.matrix('x')
        y = T.ivector('y')
        batch_size = self.batch_size

        layer0_input = x.reshape((batch_size, 1, 28, 28))
        cost = self.negative_log_likelihood(layer0_input, y)

        params = self.layer0.params + self.layer1.params + self.layer2.params + self.layer3.params
        grads = T.grad(cost, params)
        updates = [(param, param - alpha * grad) for param, grad in zip(params, grads)]

        index = T.lscalar()
        train_func = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_x[index * batch_size:(index + 1) * batch_size],
                y: train_y[index * batch_size:(index + 1) * batch_size],
            }
        )
        best_loss = self.run_batches(
            train_x,
            train_y,
            test_x,
            test_y,
            valid_x,
            valid_y,
            x,
            y,
            train_model_func=train_func,
            batch_size=batch_size,
            n_epochs=n_epochs
        )
        return best_loss
