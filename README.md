# Convolutional Autoencoder, MNIST
This is yet another implementation for the classification of the MNIST handwritten digit dataset into even and odd classes.
It is implemented as part of a homework exercise for [NDM-07-05] Neural Networks - Deep Learning course
of CS Department, AUTh


## Getting Started

### Prerequisites
1. Python (3.6 or higher, preferably 3.9)
2. venv

To install them on variant Linux distributions follow the instructions below

#### Fedora
```shell
$ sudo dnf upgrade --refresh # updates installed packages and repositories metadata
$ sudo dnf install python python3-pip python3-virtualenv python3-devel
```

#### Ubuntu 
```shell
$ sudo apt update && sudo apt upgrade # updates installed packages and repositories metadata
$ sudo apt install python3 python3-pip python3.9-venv python3.9-dev # ubuntu still offers python2 in its repositories
```

### Running the application
1. Create and activate a virtual environment 
    ```shell
    $ python3.9 -m venv venv
    $ source venv/bin/activate
    ```
2. Install necessary python dependencies 
   ```shell
   $ pip install -r requirements.txt
    ```
3. Run the application 
    ```shell
   $ python3 -m autoencoder
    ```
