import numpy as np, argparse, random
from sklearn.decomposition import PCA
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    train_file_path = 'material/mnist_train.csv'

    init_type = 'p'

    train_csv = pd.read_csv(train_file_path)

    train_label = []
    train_data = []
    for i in range(train_csv.values.shape[0]):
        train_label.append(train_csv.values[i][0])
        train_data.append([1] + train_csv.values[i][1:])
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    train_data = train_data / 255.0

    pca = PCA(n_components=50)
    train_data = pca.fit_transform(train_data)

    with open("train60000dim50.txt", "w") as f:
        for i in train_data:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'
            f.write(i)

    with open("lable.txt", "w") as f:
        for i in train_label:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'
            f.write(i)