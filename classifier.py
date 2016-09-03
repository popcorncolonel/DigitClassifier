class Classifier(object):
    def pred_label(self, x):
        raise NotImplementedError

    def train(self, train_x, train_y, test_x, test_y, valid_x, valid_y, alpha=0.13, batch_size=500, n_epochs=1000):
        raise NotImplementedError
