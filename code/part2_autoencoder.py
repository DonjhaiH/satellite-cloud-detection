import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        hidden1 = 16
        hidden2 = 32

        # convolutional encoder- extract spacial features first
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, hidden1, kernel_size=3, padding=1),
            nn.ReLU(),
            # hidden1 channels come from above
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # because we added padding (1), the patch is still 9x9

        # 32 from above by whatever the patch size is
        conv_output_size = hidden2 * patch_size * patch_size

        # flatten features into vector
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, embedding_size),
        )

        # convolution decoder (reversed process)
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_size, conv_output_size),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (hidden2, patch_size, patch_size)), # include 1 because of feature dimension
            nn.Conv2d(hidden2, hidden1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden1, n_input_channels, kernel_size=3, padding=1),
        )


    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """

        # all the autencoder does is encode then decode the input tensor
        x = self.encoder_conv(batch)
        encoded = self.encoder_fc(x)

        x = self.decoder_fc(encoded)
        decoded = self.decoder_conv(x)

        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """

        # we encode then decode by calling above functoin
        decoded = self.forward(batch)

        # the loss I have put is the MSE between the input tensor and the decoded tensor
        loss = F.mse_loss(batch, decoded)
        # you can consider other possible loss functions, or add additional terms
        # to this loss.
        # for instance, could it be good to add a term that encourages sparsity
        # in the embedding?

        # log the training loss for experiment tracking purposes
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """

        # we encode then decode
        decoded = self.forward(batch)

        # the loss I have put for validation is the MSE
        # between the input tensor and the decoded tensor
        loss = F.mse_loss(batch, decoded)
        # log the validation loss for experiment tracking purposes
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        x = self.encoder_conv(x)
        return self.encoder_fc(x)
