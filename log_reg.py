import numpy as np
import random
import theano
import theano.tensor as T
from get_data import get_data
from get_data import display_img


class LogisticRegression(object):
    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, n_in, n_out):
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
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.valid_x = valid_x
        self.valid_y = valid_y

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

    def train(self, x, y, alpha=0.13, batch_size=50, n_epochs=1000):
        n_train_batches = self.train_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = self.valid_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = self.test_x.get_value(borrow=True).shape[0] // batch_size

        index = T.lscalar()

        test_model = theano.function(
            inputs=[index],
            outputs=self.errors(x, y),
            givens={
                x: self.test_x[index * batch_size:(index+1)*batch_size],
                y: self.test_y[index * batch_size:(index+1)*batch_size],
            }
        )

        validate_model = theano.function(
            inputs=[index],
            outputs=self.errors(x, y),
            givens={
                x: self.valid_x[index * batch_size:(index+1)*batch_size],
                y: self.valid_y[index * batch_size:(index+1)*batch_size],
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
                x: self.train_x[index * batch_size:(index + 1) * batch_size],
                y: self.train_y[index * batch_size:(index + 1) * batch_size],
            }
        )

        best_loss = float('inf')
        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                train_model(minibatch_index)
            if epoch % n_train_batches == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                avg_validation_loss = np.mean(validation_losses)
                print('epoch {} -> validation error: {}'.format(epoch+1, avg_validation_loss))
                if avg_validation_loss < best_loss:
                    best_loss = avg_validation_loss
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    avg_test_loss = np.mean(test_losses)
                    print('epoch {} -> test error: {}'.format(epoch+1, avg_test_loss))


def get_xy(dataset):
    x = theano.shared(
        value=np.asarray([x[0] for x in dataset], dtype=theano.config.floatX),
        borrow=True,
    )
    y = T.cast(theano.shared(
        value=np.asarray([x[1] for x in dataset], dtype=theano.config.floatX),
        borrow=True,
    ), 'int32')
    return x, y


def main():
    data = []
    for i in range(10):
        data.extend([(d, i) for d in get_data(i)])
    random.seed(1234)
    random.shuffle(data)

    training = data[:800]
    test = data[800:900]
    validation = data[900:]

    train_x, train_y = get_xy(training)
    test_x, test_y = get_xy(test)
    valid_x, valid_y = get_xy(validation)

    classifier = LogisticRegression(train_x, train_y, test_x, test_y, valid_x, valid_y, n_in=28*28, n_out=10)
    x = T.matrix('x')
    y = T.ivector('y')
    classifier.train(x, y)
    for img, label in random.sample(data, 20):
        img = np.asarray(img, dtype=theano.config.floatX)
        pred_label = classifier.pred_label(img)
        print("I think this is a {}.".format(pred_label))
        print("it is actually a {}.".format(label))
        display_img(img)


if __name__ == '__main__':
    main()