from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self, encoding_layers: list):
        """
        encoding_layers: input dimension is the first item of the list, encoding dimension is the last item of the layers list (decoder is symmetric)
        """

        super().__init__()

        # Encoding layers
        encoding = []
        for i in range(len(encoding_layers) - 1):
            encoding.append(nn.Linear(encoding_layers[i], encoding_layers[i + 1]))
            encoding.append(nn.ReLU(inplace=True))
            encoding.append(nn.BatchNorm1d(encoding_layers[i + 1]))

        # Decoding layers
        decoding_layers = encoding_layers[::-1]
        decoding = []
        for i in range(len(decoding_layers) - 1):
            decoding.append(nn.Linear(decoding_layers[i], decoding_layers[i + 1]))
            decoding.append(nn.ReLU(inplace=True))
            decoding.append(nn.BatchNorm1d(decoding_layers[i + 1]))

        self.encoder = nn.Sequential(*encoding)
        self.decoder = nn.Sequential(*decoding)

    def forward(self, x):
        embeddings = []
        for layer in self.encoder:
            x = layer(x)
            embeddings.append(x)

        x_encoded = embeddings[-1]
        x_reconstructed = self.decoder(x_encoded)
        return x_encoded, x_reconstructed, embeddings
