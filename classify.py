import numpy as np
import random
import theano

from get_data import get_data
from get_data import display_img
from get_data import get_xy
from mlp import MLP

data = []
random.seed(1234)
for i in range(10):
    data.extend([(d, i) for d in get_data(i)])
random.shuffle(data)

training = data[:8000]
test = data[8000:9000]
validation = data[9000:]

train_x, train_y = get_xy(training)
test_x, test_y = get_xy(test)
valid_x, valid_y = get_xy(validation)


def optimize_hyperparam(classifier, hyperparam_name, possible_range=(-999, 999)):
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
    for val in possible_vals:
        print('Training with {name} as val... {val}'.format(name=hyperparam_name, val=best_val, error=best_error))
        error = classifier.train(train_x, train_y, test_x, test_y, valid_x, valid_y, **{hyperparam_name: val})
        if error < best_error:
            best_error = error
            best_val = val

    print('{name} optimal val: {val} with error {error}'.format(name=hyperparam_name, val=best_val, error=best_error))
    return best_val


def main():

    classifier = MLP(n_in=28*28, n_hidden=500, n_out=10, rng=np.random.RandomState(1234))
    best_alpha = optimize_hyperparam(classifier, 'alpha', (0.001, 0.2))
    best_l1 = optimize_hyperparam(classifier, 'l1_reg', (0.000, 0.5))
    best_l2 = optimize_hyperparam(classifier, 'l2_reg', (0.000, 0.5))
    best_batch_size = optimize_hyperparam(classifier, 'batch_size', (20, 1000))

    classifier.train(train_x, train_y, test_x, test_y, valid_x, valid_y, alpha=best_alpha, l1_reg=best_l1, l2_reg=best_l2, batch_size=best_batch_size)
    #classifier.train(train_x, train_y, test_x, test_y, valid_x, valid_y)

    for img, label in random.sample(data, 20):
        img = np.asarray(img, dtype=theano.config.floatX)
        pred_label = classifier.pred_label(img)
        print("Guessed: {}; Actually: {}".format(pred_label, label))
        display_img(img)


if __name__ == '__main__':
    main()
