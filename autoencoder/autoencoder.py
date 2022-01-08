from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, dropout_rate):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5),
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
            nn.Linear(3200, 10)
        )

        self.softmax = nn.Softmax(dim=1)

        self.decoder = nn.Sequential(
            nn.Linear(10, 400),
            nn.ReLU(True),
            nn.Unflatten(1, (1, 20, 20)),
            nn.Dropout2d(p=dropout_rate),
            nn.ConvTranspose2d(1, 10, kernel_size=5),
            nn.ReLU(True),
            nn.Dropout2d(p=dropout_rate),
            nn.ConvTranspose2d(10, 1, kernel_size=5)

        )

    def forward(self, x):
        enc = self.encoder(x)
        return enc, self.decoder(self.softmax(enc))
