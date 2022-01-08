import numpy as np
import torch
from torchvision import transforms
from autoencoder.utils import download_mnist_dataset, visualize_batch, visualize_image, visualize_after_decode, train, evaluate, stopwatch
from autoencoder.autoencoder import AutoEncoder
from torch import nn
import knn_classifier as knn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0
    pin_memory = False

    if device == "cuda":
        num_workers = 4
        pin_memory = True
    print(f"Will be using {device} for training and testing")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    batch_size = 64
    n_epochs = 30

    train_set = download_mnist_dataset('mnist', True, transform=transform)
    test_set = download_mnist_dataset('mnist', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=num_workers,
                                               pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=num_workers,
                                              pin_memory=pin_memory)
    # visualizes a part of the batch
    visualize_batch(train_loader)
    # visualizes an image in more detail
    visualize_image(np.squeeze((next(iter(train_loader))[0])[0].numpy()))
    encoder = AutoEncoder()
    distance = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    mse_multiplier = 0.5
    cel_multiplier = 0.5
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.002)
    images, labels = next(iter(train_loader))
    images = images.view(images.shape[0], -1)
    encoder.to(device)
    multipliers: tuple = (mse_multiplier, cel_multiplier)
    print(
        f"Training took {stopwatch(lambda: train(model=encoder, device=device, train_loader=train_loader, distance=distance, criterion=criterion, optimizer=optimizer, epochs=n_epochs))}s")
    print(
        f"Evaluation took {stopwatch(lambda: evaluate(model=encoder, device=device, test_loader=test_loader, criterion=criterion))}s")
    visualize_after_decode(model=encoder, device=device, test_loader=test_loader)

    # transform data to numpy arrays
    train_df = train_set.data.numpy()
    train_label = train_set.targets.numpy()
    train_df = np.reshape(train_df, (train_df.shape[0], train_df.shape[1] * train_df.shape[2]))

    test_df = test_set.data.numpy()
    test_label = test_set.targets.numpy()
    test_df = np.reshape(test_df, (test_df.shape[0], test_df.shape[1] * test_df.shape[2]))

    for k in range(1, 4, 2):
        print(f"Running KNN classification for {k} nearest neighbor(s)\n")
        knn.knn_classification(train_df, train_label, test_df, test_label, k)

    print(f"Running K-nearest centroid classification\n")
    knn.centroid_classification(train_df, train_label, test_df, test_label)
