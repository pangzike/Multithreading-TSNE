import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.loadtxt('output.txt')
    y = np.loadtxt('label.txt')[0:x.shape[0]]
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()


if __name__ == '__main__':
    main()
