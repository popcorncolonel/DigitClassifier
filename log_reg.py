import numpy as np
import theano
import theano.tensor as T

from classifier import Classifier


class LogisticRegression(Classifier):
    def __init__(self, n_in, n_out):
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX,
            ),
            name='W',
            borrow=True,
        )
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX,
            ),
            name='b',
            borrow=True,
        )
        self.params = [self.W, self.b]

    def p_y_given_x(self, in_vect):
        return T.nnet.softmax(T.dot(in_vect, self.W) + self.b)

    def predict(self, x):
        return T.argmax(self.p_y_given_x(x), axis=1)

    def negative_log_likelihood(self, x, y):
        n_examples_in_batch = y.shape[0]
        log_probs = T.log(self.p_y_given_x(x))
        return -T.mean(log_probs[(T.arange(n_examples_in_batch), y)])

    def pred_label(self, x):
        x_ = T.vector('x')
        return theano.function(
            inputs=[],
            outputs=self.predict(x_),
            givens={
                x_: x
            }
        )()[0]

    def errors(self, x, y):
        y_pred = self.predict(x)
        assert y.ndim == y_pred.ndim
        if y.dtype.startswith('int'):
            equalities = T.neq(y, y_pred)
            return T.mean(equalities)

    def train(self, train_x, train_y, test_x, test_y, valid_x, valid_y, alpha=0.13, batch_size=500, l1_reg=0., l2_reg=0., n_epochs=1000):
        batch_size = int(batch_size)
        n_epochs = 1000

        x = T.matrix('x')
        y = T.ivector('y')

        n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size

        index = T.lscalar()

        test_model = theano.function(
            inputs=[index],
            outputs=self.errors(x, y),
            givens={
                x: test_x[index * batch_size:(index+1)*batch_size],
                y: test_y[index * batch_size:(index+1)*batch_size],
            }
        )

        validate_model = theano.function(
            inputs=[index],
            outputs=self.errors(x, y),
            givens={
                x: valid_x[index * batch_size:(index+1)*batch_size],
                y: valid_y[index * batch_size:(index+1)*batch_size],
            }
        )

        cost = self.negative_log_likelihood(x, y)

        g_W = T.grad(cost=cost, wrt=self.W)
        g_b = T.grad(cost=cost, wrt=self.b)
        updates = [
            (self.W, self.W - alpha * g_W),
            (self.b, self.b - alpha * g_b),
        ]

        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_x[index * batch_size:(index + 1) * batch_size],
                y: train_y[index * batch_size:(index + 1) * batch_size],
            }
        )

        best_loss = float('inf')
        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                train_model(minibatch_index)
            if epoch % n_train_batches == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                avg_validation_loss = np.mean(validation_losses)
                print('epoch {} -> validation error: {} (best loss={})'.format(epoch+1, avg_validation_loss, best_loss))
                if avg_validation_loss < best_loss:
                    best_loss = avg_validation_loss
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    avg_test_loss = np.mean(test_losses)
                    print('epoch {} -> test error: {}'.format(epoch+1, avg_test_loss))
        return best_loss

