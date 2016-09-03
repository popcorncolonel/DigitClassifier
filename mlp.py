import numpy as np
import theano
import theano.tensor as T
from classifier import Classifier
from log_reg import LogisticRegression


class HiddenLayer(object):
    def __init__(self, n_in, n_out, rng, activation=T.tanh):
        self.W = theano.shared(
            value=np.asarray(
                rng.uniform(
                    low=-np.sqrt(6 / (n_in+n_out)),
                    high=np.sqrt(6 / (n_in+n_out)),
                    size=(n_in, n_out),
                ),
                dtype=theano.config.floatX,
            ),
            name='W',
            borrow=True,
        )
        self.b = theano.shared(
            value=np.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True,
        )
        self.activation = activation
        self.params = [self.W, self.b]

    def output(self, in_vect):
        lin_output = T.dot(in_vect, self.W) + self.b
        return self.activation(lin_output)


class MLP(Classifier):
    def __init__(self, n_in, n_hidden, n_out, rng):
        self.hidden = HiddenLayer(
            n_in=n_in,
            n_out=n_hidden,
            rng=rng,
        )
        self.log_reg = LogisticRegression(
            n_in=n_hidden,
            n_out=n_out,
        )
        self.l1_norm = abs(self.hidden.W).sum() + abs(self.log_reg.W).sum()
        self.l2_norm_square = (self.hidden.W ** 2).sum() + (self.log_reg.W ** 2).sum()
        self.params = self.hidden.params + self.log_reg.params

    def negative_log_likelihood(self, x, y):
        hidden_output = self.hidden.output(x)
        return self.log_reg.negative_log_likelihood(hidden_output, y)

    def errors(self, x, y):
        hidden_output = self.hidden.output(x)
        return self.log_reg.negative_log_likelihood(hidden_output, y)

    def pred_label(self, x):
        hidden_output = self.hidden.output(x)
        return self.log_reg.pred_label(hidden_output)

    def train(self, train_x, train_y, test_x, test_y, valid_x, valid_y, alpha=0.13, batch_size=500):
        x = T.matrix('x')
        y = T.ivector('y')

        lambda_1 = 1
        lambda_2 = 1
        cost = (
            self.negative_log_likelihood(x, y) +
            lambda_1 * self.l1_norm +
            lambda_2 * self.l2_norm_square
        )
        index = T.lscalar()

        grads = [T.grad(cost, param) for param in self.params]
        updates = [(param, param - alpha*grad) for param, grad in zip(self.params, grads)]
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_x[index*batch_size:(index+1)*batch_size],
                y: train_y[index*batch_size:(index+1)*batch_size],
            }
        )
        self.run_batches(train_x, train_y, test_x, test_y, valid_x, valid_y, x, y, train_model)

