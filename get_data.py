import numpy as np
import matplotlib.pyplot as plt


def get_data(num: int) -> list:
    """
    Returns a list of examples of images
    """
    images = []
    with open('data/data' + str(num), 'rb') as f:
        for _ in range(1000):
            img = []
            for i in range(28*28):
                byte = f.read(1)
                img.append(byte[0])
            images.append(np.asarray(img))
    return images


def display_img(arr: np.array):
    arr = arr.reshape((28, 28))
    plt.imshow(arr, cmap='Greys_r')
    plt.show()
