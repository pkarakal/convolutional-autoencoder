import time

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from autoencoder.utils import download_mnist_dataset
import numpy as np
from sklearn.metrics import classification_report


def knn_classification(train_set, train_labels, test_set, test_labels, k=3):
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_set, train_labels)
    print(f"Training Time: {(time.time() - start_time)} seconds.")
    start_time = time.time()
    pred = knn.predict(test_set)
    print(f"Prediction Time: {(time.time() - start_time)}s seconds.")
    report_knn = classification_report(test_labels, pred)
    print(report_knn)


def centroid_classification(train_set, train_labels, test_set, test_labels):
    start_time = time.time()
    nc = NearestCentroid()
    nc.fit(train_set, train_labels)
    print(f"Training Time: {(time.time() - start_time)} seconds.")
    pred = nc.predict(test_set)
    print(f"Prediction Time: {(time.time() - start_time)}s seconds.")
    report_nc = classification_report(test_labels, pred)
    print(report_nc)


def main():
    # load mnist dataset using pytorch and convert it to ndarrays
    train_dataset = download_mnist_dataset('mnist', True, transform=None)
    train_df = train_dataset.data.numpy()
    train_label = train_dataset.targets.numpy()

    train_df = np.reshape(train_df, (train_df.shape[0], train_df.shape[1] * train_df.shape[2]))

    test_dataset = download_mnist_dataset('mnist', False, transform=None)
    test_df = test_dataset.data.numpy()
    test_label = test_dataset.targets.numpy()

    test_df = np.reshape(test_df, (test_df.shape[0], test_df.shape[1] * test_df.shape[2]))

    for k in range(1, 4, 2):
        knn_classification(train_df, train_label, test_df, test_label, k)

    centroid_classification(train_df, train_label, test_df, test_label)


if __name__ == "__main__":
    main()
