import numpy as np
import random
import theano
import theano.tensor as T


class Classifier(object):
    def errors(self, x, y):
        raise NotImplementedError

    def pred_label(self, x):
        raise NotImplementedError

    def train(self, train_x, train_y, test_x, test_y, valid_x, valid_y, alpha=0.13, batch_size=500, l1_reg=0., l2_reg=0.):
        raise NotImplementedError

    def run_batches(self, train_x, train_y, test_x, test_y, valid_x, valid_y, x, y, train_model_func, batch_size=500, n_epochs=1000):
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

        best_loss = float('inf')
        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                train_model_func(minibatch_index)
            if epoch % 5 == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                avg_validation_loss = np.mean(validation_losses)
                print('epoch {} -> validation error: {} (best loss={})'.format(epoch, avg_validation_loss, best_loss))
                if avg_validation_loss < best_loss:
                    best_loss = avg_validation_loss
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    avg_test_loss = np.mean(test_losses)
                    print('epoch {} -> test error: {}'.format(epoch+1, avg_test_loss))
        return best_loss
