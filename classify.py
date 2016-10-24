import numpy as np
import pickle
import random
import timeit

from get_data import get_data
from get_data import display_img
from get_data import get_xy
from mlp import MLP
from cnn import ConvolutionalNeuralNetwork

random.seed(1234)
print('loading data...')
training, valid, test = get_data()
print('loaded.')

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
    best_alpha = 0.02
    best_l1 = 0.000
    best_l2 = 0.001
    best_batch_size = 200
    n_epochs = 1000

    rng = np.random.RandomState(1234)
    """
    #classifier = MLP(n_in=28*28, n_hidden=500, n_out=10, rng=rng)
    classifier = ConvolutionalNeuralNetwork(
        rng=rng,
        batch_size=best_batch_size,
        nkerns=(20, 50),
    )

    print("training")
    start_time = timeit.default_timer()
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
    end_time = timeit.default_timer()
    print('Trained for %.1fs' % (end_time - start_time))
    """

    with open('best_model.pkl', 'r') as f:
        best_model = pickle.load(f)
    n_correct = 0
    n_wrong = 0
    best_model.batch_size = 1
    for img, label in zip(test[0], test[1]):
        img = np.asarray(img)
        predicted = best_model.pred_label(img)
        if predicted != label:
            print('guessed {} but was actually {}'.format(predicted, label))
            #display_img(img)
            n_wrong += 1
        else:
            print('correctly guessed {}'.format(predicted))
            n_correct += 1
    print("{} correct, {} incorrect".format(n_correct, n_wrong))


if __name__ == '__main__':
    main()
