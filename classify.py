import numpy as np
import random
import theano

from get_data import get_data
from get_data import display_img
from get_data import get_xy
from mlp import MLP

random.seed(1234)
training, valid, test = get_data()

train_x, train_y = get_xy(training[0], training[1])
test_x, test_y = get_xy(test[0], test[1])
valid_x, valid_y = get_xy(valid[0], valid[1])


def optimize_hyperparam(classifier, hyperparam_name, possible_range=(-999, 999), **kwargs):
    best_val = possible_range[0]
    best_error = float('inf')
    diff = possible_range[1] - possible_range[0]
    val = possible_range[0]
    possible_vals = []
    num_to_try = 10.0

    for _ in range(int(num_to_try)):
        possible_vals.append(val)
        val += diff / num_to_try

    possible_vals.append(val)
    kwargs.update({hyperparam_name: val})
    for val in possible_vals:
        print('Training with {name} as val... {val}'.format(name=hyperparam_name, val=val))
        error = classifier.train(train_x, train_y, test_x, test_y, valid_x, valid_y, **kwargs)
        if error < best_error:
            best_error = error
            best_val = val

    print('{name} optimal val: {val} with error {error}'.format(name=hyperparam_name, val=best_val, error=best_error))
    return best_val, best_error


def main():
    classifier = MLP(n_in=28*28, n_hidden=500, n_out=10, rng=np.random.RandomState(1234))
    best_alpha = 0.0806
    best_l1 = 0.000
    best_l2 = 0.001
    best_batch_size = 600
    n_epochs = 1000

    classifier.train(
        train_x,
        train_y,
        test_x,
        test_y,
        valid_x,
        valid_y,
        alpha=best_alpha,
        l1_reg=best_l1,
        l2_reg=best_l2,
        batch_size=best_batch_size,
        n_epochs=n_epochs
    )

    print('best hyperparams: {}'.format({
        'alpha': best_alpha,
        'l1': best_l1,
        'l2': best_l2,
        'batch_size': best_batch_size,
    }))


if __name__ == '__main__':
    main()
