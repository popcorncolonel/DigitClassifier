import numpy as np
import random
import theano

from get_data import get_data
from get_data import display_img
from get_data import get_xy
from log_reg import LogisticRegression


def main():
    data = []
    for i in range(10):
        data.extend([(d, i) for d in get_data(i)])
    random.shuffle(data)

    training = data[:8000]
    test = data[8000:9000]
    validation = data[9000:]

    train_x, train_y = get_xy(training)
    test_x, test_y = get_xy(test)
    valid_x, valid_y = get_xy(validation)

    classifier = LogisticRegression(n_in=28*28, n_out=10)
    classifier.train(train_x, train_y, test_x, test_y, valid_x, valid_y)

    for img, label in random.sample(data, 20):
        img = np.asarray(img, dtype=theano.config.floatX)
        pred_label = classifier.pred_label(img)
        print("Guessed: {}; Actually: {}".format(pred_label, label))
        display_img(img)


if __name__ == '__main__':
    main()
