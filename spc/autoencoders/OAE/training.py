import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


##############################
#                            #
#       Early Stopping       #
#                            #
##############################

class EarlyStopping:
    """Stops the training when the validation loss does not improve more than a given delta."""

    def __init__(self, delta=0, patience=10, path='best_model.pt', verbose=True):
        """
        :param delta: minimum desired improvement
        :param patience: number of successive iterations for which delta improvement must be observed
        :param path: where to save best model
        :param verbose: talkative
        """
        self.delta = delta
        self.patience = patience
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_best_model(val_loss, model)

        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_best_model(val_loss, model)
            self.counter = 0

    def save_best_model(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss


##############################
#                            #
#          Training          #
#                            #
##############################

class TrainingModel:
    """Prepare data and train network."""

    def __init__(self, penalty=1, learning_rate=1e-3, batch_size=50, val_size=0.20, verbose=True):
        """
        :param penalty: importance given to the orthogonality regularization term
        :param learning_rate: to update network parameters when training
        :param batch_size: number of observations processed together before updating parameters
        :param val_size: percentage of data to be allocated to validation to avoid overfitting
        :param verbose: talkative
        """
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.val_size = val_size
        self.verbose = verbose

    def custom_loss(self, compression, reconstruction, original):
        loss = torch.mean((reconstruction - original) ** 2) + \
               self.penalty * torch.mean(
            (torch.matmul(compression.T, compression) - torch.eye(compression.shape[1])) ** 2)
        return loss

    def create_datasets(self, train_data):

        # obtain training indices that will be used for validation
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.val_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # load training data in batches
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler,
                                                        num_workers=0)

        # load validation data in batches
        self.valid_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=valid_sampler,
                                                        num_workers=0)

        return self.train_loader, self.valid_loader

    def train_model(self, model, patience=10, n_epochs=100, path='best_model.pt'):

        history = dict(train=[], val=[])
        early_stopping = EarlyStopping(patience=patience, verbose=self.verbose)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        for epoch in range(1, n_epochs + 1):

            # Training
            train_losses = []
            model.train()  # prep model for training
            for batch, data in enumerate(self.train_loader, 1):
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                encoded, decoded, _ = model(data)
                # calculate the loss
                loss = self.custom_loss(compression=encoded, reconstruction=decoded, original=data)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # record training loss
                train_losses.append(loss.item())

            # Validation
            val_losses = []
            model.eval()  # prep model for evaluation
            with torch.no_grad():
                for data in self.valid_loader:
                    # forward pass: compute predicted outputs by passing inputs to the model
                    encoded, decoded, _ = model(data)
                    # calculate the loss
                    loss = self.custom_loss(compression=encoded, reconstruction=decoded, original=data)
                    # record validation loss
                    val_losses.append(loss.item())

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            val_loss = np.average(val_losses)
            history['train'].append(train_loss)
            history['val'].append(val_loss)

            epoch_len = len(str(n_epochs))

            if self.verbose:
                print(f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' + f'train_loss: {train_loss:.6f} ' +
                      f'valid_loss: {val_loss:.6f}')

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                if self.verbose:
                    print(f"-----------> Early stopping: no improvement after {patience} consecutive epochs")
                break

        model = torch.load(path)

        return model.eval(), history
