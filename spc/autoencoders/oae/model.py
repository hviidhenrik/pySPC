from torch import nn
import torch
import torch.nn.functional as F


class AutoEncoder(nn.Module):

    def __init__(self, encoding_layers, penalty=0.1, norm_type="l2"):
        """
        :param encoding_layers: input dimension is the first item of the list, encoding dimension is the last item of the layers list (decoder is symmetric)
        :param penalty
        :param norm_type: either "frobenius" or "l2"
        """

        super().__init__()
        self.penalty = penalty
        self.norm_type = norm_type

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
        return x_encoded, x_reconstructed

    def loss_function(self, compression, reconstruction, original):
        """
        Ortho-loss to learn uncorrelated features. The penalty term computed as the Frobenius or L2 norm of the difference
        between the dot product of the encoded features and the identity matrix may be negative if the dot product is
        greater than the identity matrix. By squaring the penalty term, we ensure that it is always positive and has a
        greater impact on the overall loss function, which encourages the encoded features to be more orthogonal.
        :param compression: encoded features from the bottleneck
        :param reconstruction: reconstructed features from the last layer
        :param original: input features
        :return: loss value
        """
        mse_loss = F.mse_loss(reconstruction, original)

        # Compute penalty term
        dot_product = torch.matmul(compression.T, compression)
        if self.norm_type == 'frobenius':
            penalty_loss = torch.norm(dot_product - torch.eye(dot_product.shape[0]), p='fro') ** 2
        elif self.norm_type == 'l2':
            penalty_loss = torch.norm(dot_product - torch.eye(dot_product.shape[0]), p=2) ** 2
        else:
            raise ValueError("Invalid norm type: {}, use either frobenius or l2 norms".format(self.norm_type))
        loss = mse_loss + self.penalty * penalty_loss
        return loss
