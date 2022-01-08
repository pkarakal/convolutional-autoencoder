import time

import torch.nn.functional
from torch import max, no_grad
from torchvision.datasets import MNIST
from os import path
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import softmax


def download_mnist_dataset(root: str = 'data', train: bool = True, transform=None) -> MNIST:
    return MNIST(root, train=train, download=not (path.exists(root) and path.isdir(root)), transform=transform)


def visualize_batch(train_loader):
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    images = images.numpy()

    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx+1)
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(str(labels[idx].item()))

    plt.show()


def visualize_image(img):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(str(val), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')
    plt.show()


def train(model, device, train_loader, distance, criterion, optimizer, epochs=50,
          multipliers: tuple = (0.5, 0.5)):
    model.train()  # prep model for training

    mse_multiplier = multipliers[0]
    cel_multiplier = multipliers[1]
    for epoch in range(epochs):
        # monitor training loss
        mse_loss = 0.0
        cel_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            out_enc, output = model(images)
            # calculate the loss
            loss_mse = distance(output, images)
            loss_cel = criterion(out_enc, labels)
            loss = (mse_multiplier * loss_mse) + (cel_multiplier * loss_cel)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            mse_loss += loss_mse.item()
            cel_loss += loss_cel.item()

        print('Epoch: {} \tMSE Loss: {:.6f} \t Cross Entropy Loss {:.6}'.format(
            epoch + 1,
            mse_loss/len(train_loader),
            cel_loss/len(train_loader)
        ))
    return


def evaluate(model, device, test_loader, criterion):
    test_loss = 0.0
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    model.eval()  # prep model for *evaluation*

    with no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # data = data.view(data.shape[0], -1)
            output, _ = model(data)
            prob = softmax(output, dim=1)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(prob, dim=1)
            pred.to(device)
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            for i in range(16):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    print('Average Test Loss: {:.6f}\n'.format(test_loss))
    for i in range(len(class_total)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    return


def stopwatch(func):
    start = time.time()
    func()
    end = time.time()
    return f"{end - start}"


def visualize_after_decode(model, test_loader, device):
    model.eval()
    img, labels = list(test_loader)[0]
    img = img.to(device)
    _, output = model(img)
    inp = img[0:10, 0, :, :].squeeze().detach().cpu()
    out = output[0:10, 0, :, :].squeeze().detach().cpu()

    inp = inp.permute(1, 0, 2).reshape(28, -1).numpy()
    out = out.permute(1, 0, 2).reshape(28, -1).numpy()
    combined = np.vstack([inp, out])

    plt.imshow(combined)
    plt.show()

