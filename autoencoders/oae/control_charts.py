import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from autoencoders.oae.model import AutoEncoder
from autoencoders.oae.training import TrainingModel
from autoencoders.oae.utils import estimate_upper_control_limit, plot_multivariate_control_charts, integrated_gradients


class ProcessMonitoring:

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.mean_vector_original_data = None
        self.columns = None
        self.model = None
        self.trained_model = None
        self.losses = None
        self.x_encoded_phase1 = None
        self.inverted_cov_matrix = None
        self.mean_vector = None
        self.t2_scores_phase1 = None
        self.ucl_t2 = None
        self.residuals_phase1 = None
        self.ucl_spe = None
        self.t2_scores_phase2 = None
        self.residuals_phase2 = None

    def fit(self, data, encoding_layers, orthogonality_regularization=0.1, norm="fro", nr_epochs=100, patience=10, batch_size=100, verbose=True):
        """
        training autoencoders and finding upper control limits
        :param data: phase 1 data, pandas DataFrame
        :param encoding_layers: layers of the encoder newtork (decoder is symmetrical), e.g., [input_dim, hidden_dim, ..., encoding_dim]
        :param orthogonality_regularization: float value
        :param norm: norm type, can be "fro" (Frobenius), "l1" or "l2"
        :param nr_epochs: max number of epochs
        :param patience: for early stopping, number of epochs without improvement tolerated
        :param batch_size: int
        :param verbose: whether to print the training history or not
        :return:
        """
        # standardizing phase one data
        data = pd.DataFrame(self.scaler.fit_transform(data))
        self.mean_vector_original_data = np.mean(data, axis=0)

        # initializing model
        self.model = AutoEncoder(encoding_layers=encoding_layers, penalty=orthogonality_regularization, norm_type=norm)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        x = torch.Tensor(data.values)

        # training model
        trainer = TrainingModel(learning_rate=1e-3, batch_size=batch_size, val_size=0.20, verbose=verbose)
        trainer.create_datasets(train_data=x)
        self.trained_model, self.losses = trainer.train_model(self.model, patience=patience, n_epochs=nr_epochs)

        # obtaining encoded and reconstructed features
        self.x_encoded_phase1, x_reconstructed = self.trained_model(x)
        self.x_encoded_phase1 = self.x_encoded_phase1.cpu().detach().numpy()
        x_reconstructed = pd.DataFrame(x_reconstructed.cpu().detach().numpy())

        # Hotelling T^2 control chart
        self.inverted_cov_matrix = np.linalg.inv(np.cov(self.x_encoded_phase1.T))
        self.mean_vector = np.mean(self.x_encoded_phase1, axis=0)
        self.t2_scores_phase1 = np.array([(sample - self.mean_vector).T @ self.inverted_cov_matrix @ (sample - self.mean_vector) for sample in self.x_encoded_phase1])
        self.ucl_t2 = estimate_upper_control_limit(self.t2_scores_phase1)

        # SPE control chart
        data.reset_index(drop=True, inplace=True)
        self.columns = np.arange(0, data.shape[1])
        self.residuals_phase1 = ((data.subtract(x_reconstructed, axis=1)) ** 2).sum(axis=1)
        self.ucl_spe = estimate_upper_control_limit(self.residuals_phase1)

        print(f"Training complete! Upper control limits are: {self.ucl_t2:4.2f} (T^2) and {self.ucl_spe:.2f} (SPE)")

    def plot_phase1(self, log_scale=False):
        """
        finds UCLs and plots control charts
        :param log_scale: whether to plot the values on a log scale
        :return: classification into in-control and out-of-control with T^2 and SPE
        """
        plot_multivariate_control_charts(t2_scores=self.t2_scores_phase1, t2_ucl=self.ucl_t2,
                                         spe_scores=self.residuals_phase1, spe_ucl=self.ucl_spe, log_scale=log_scale)
        t2_out_of_control = self.t2_scores_phase1 > self.ucl_t2
        spe_out_of_control = self.residuals_phase1 > self.ucl_spe

        return {"T2": t2_out_of_control.astype(int), "SPE": spe_out_of_control.astype(int)}

    def plot_phase2(self, data, log_scale=False):
        """
        finds UCLs and plots control charts
        :param data: phase 2 data
        :param log_scale: whether to plot the values on a log scale
        :return: classification into in-control and out-of-control with T^2 and SPE
        """
        data = pd.DataFrame(self.scaler.transform(data))
        x = torch.Tensor(data.values)

        x_encoded, x_reconstructed = self.trained_model(x)
        x_encoded = x_encoded.cpu().detach().numpy()
        x_reconstructed = pd.DataFrame(x_reconstructed.cpu().detach().numpy())
        self.t2_scores_phase2 = np.array([(sample - self.mean_vector).T @ self.inverted_cov_matrix @ (sample - self.mean_vector) for sample in x_encoded])
        self.residuals_phase2 = ((data.subtract(x_reconstructed, axis=1)) ** 2).sum(axis=1)

        plot_multivariate_control_charts(t2_scores=self.t2_scores_phase2, t2_ucl=self.ucl_t2,
                                         spe_scores=self.residuals_phase2, spe_ucl=self.ucl_spe, log_scale=log_scale)
        t2_out_of_control = self.t2_scores_phase2 > self.ucl_t2
        spe_out_of_control = self.residuals_phase2 > self.ucl_spe

        return {"T2": t2_out_of_control.astype(int), "SPE": spe_out_of_control.astype(int)}

    def plot_correlation_matrix(self):
        """
        Plot the covariance matrix of the encoded features from phase 1
        """
        # Compute the covariance matrix of the encoded features
        cov = np.corrcoef(self.x_encoded_phase1.T)

        # Determine step size for ticks
        n_ticks = len(cov)
        step = max(1, n_ticks // 10)

        # Plot the covariance matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cov, cmap="viridis")
        ax.set_xticks(np.arange(0, n_ticks, step))
        ax.set_yticks(np.arange(0, n_ticks, step))
        ax.set_xticklabels(np.arange(1, n_ticks + 1, step))
        ax.set_yticklabels(np.arange(1, n_ticks + 1, step))
        ax.set_title("Correlation Matrix of Encoded Features")
        fig.colorbar(im)
        plt.show()

    def t2_contribution_plots(self, ooc_observation, method="monte_carlo", samples=1000, steps=20):
        gradient_scores = integrated_gradients(model=self.trained_model, inp=ooc_observation, baseline=self.mean_vector_original_data,
                                               approximation_method=method, steps=steps, samples=samples)
        plt.bar(self.columns, gradient_scores[0], color="c", alpha=0.6)
        plt.axhline(y=0, c="k", lw=0.5)
        plt.title("$T^2$ Contribution Plots")
        plt.xlabel("Variables")
        plt.ylabel("Integrated gradients")
        plt.show()

    def spe_contribution_plots(self, ooc_observation):
        data = pd.DataFrame(self.scaler.transform(np.array(ooc_observation).reshape(1, -1)))
        x = torch.Tensor(data.values)
        _, x_reconstructed = self.trained_model(x.reshape(1, -1))
        x_reconstructed = x_reconstructed.cpu().detach().numpy()
        residuals = np.array(x - x_reconstructed) ** 2
        plt.bar(self.columns, residuals[0], color="r", alpha=0.6)
        plt.title("SPE Contribution Plots")
        plt.xlabel("Variables")
        plt.ylabel("Squared residuals")
        plt.show()
