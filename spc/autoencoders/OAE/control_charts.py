import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from scipy.stats import gaussian_kde, norm
from scipy.optimize import brentq
from model import AutoEncoder
from training import TrainingModel


class ProcessMonitoring:

    def __init__(self, alfa=0.95):
        self.alfa = alfa

    def fit(self, data, encoding_layers, orthogonality_regularization, nr_epochs, patience, batch_size, verbose):

        # standardizing phase one data
        self.scaler = StandardScaler()
        data = pd.DataFrame(self.scaler.fit_transform(data))

        # initializing model
        self.model = AutoEncoder(encoding_layers=encoding_layers)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        x = torch.Tensor(data.values)

        # training model
        trainer = TrainingModel(penalty=orthogonality_regularization, learning_rate=1e-3, batch_size=batch_size, val_size=0.20, verbose=verbose)
        trainer.create_datasets(train_data=x)
        self.trained_model, train_losses, valid_losses = trainer.train_model(self.model, patience=patience, n_epochs=nr_epochs)

        # obtaining encoded and reconstructed features
        x_encoded, x_reconstructed, _ = self.trained_model(x)
        x_encoded = x_encoded.cpu().detach().numpy()
        x_reconstructed = pd.DataFrame(x_reconstructed.cpu().detach().numpy())

        # Hotelling T^2 control chart
        self.inverted_cov_matrix = np.linalg.inv(np.cov(x_encoded.T))
        self.mean_vector = np.mean(x_encoded, axis=0)
        self.t2_scores_phase1 = np.array([(sample - self.mean_vector).T @ self.inverted_cov_matrix @ (sample - self.mean_vector) for sample in x_encoded])
        band_width = gaussian_kde(self.t2_scores_phase1).covariance_factor() * self.t2_scores_phase1.std()
        self.ucl_t2 = brentq(lambda s: sum(norm.cdf((s - self.t2_scores_phase1) / band_width)) / len(self.t2_scores_phase1) - self.alfa, 0, 100000)

        # SPE control chart
        data.reset_index(drop=True, inplace=True)
        data.columns = np.arange(0, data.shape[1])
        self.residuals_phase1 = ((data.subtract(x_reconstructed, axis=1)) ** 2).sum(axis=1)
        band_width = gaussian_kde(self.residuals_phase1).covariance_factor() * self.residuals_phase1.std()
        self.ucl_spe = brentq(lambda s: sum(norm.cdf((s - self.residuals_phase1) / band_width)) / len(self.residuals_phase1) - self.alfa, 0, 100000)

        print(f"Training complete! Upper control limits are: {self.ucl_t2:4:3f} (T^2) and {self.ucl_spe:4:3f} (SPE)")

    def plot_phase1(self, log_scale=False):
        if log_scale:
            self.t2_scores_phase1 = np.log(self.t2_scores_phase1)
            self.ucl_t2 = np.log(self.ucl_t2)
            self.residuals_phase1 = np.log(self.residuals_phase1)
            self.ucl_spe = np.log(self.ucl_spe)

        fig, axs = plt.subplots(2, 1, figsize=(16, 8), dpi=300)

        # T^2 chart
        axs[0].plot(self.t2_scores_phase1, marker='', color='black', linewidth=0.8, alpha=1)
        axs[0].axhline(y=self.ucl_t2, color='red', label="UCL", alpha=0.6)
        axs[0].grid(False)
        axs[0].set_yticks(color='black', fontsize=13)
        axs[0].set_xticks(color='black', fontsize=13)
        axs[0].set_ylabel("$T^2$ statistic", fontsize=14)
        axs[0].set_xlabel("Observations", fontsize=14)
        axs[0].legend(loc="upper left", fontsize=12)
        axs[0].set_title("Control Chart for Phase 1 Data - $T^2$", loc='center', fontsize=14, fontweight=1)

        # SPE chart
        axs[1].plot(self.residuals_phase1, marker='', color='black', linewidth=0.8, alpha=1)
        axs[1].axhline(y=self.ucl_spe, color='red', label="UCL", alpha=0.6)
        axs[1].grid(False)
        axs[1].set_yticks(color='black', fontsize=13)
        axs[1].set_xticks(color='black', fontsize=13)
        axs[1].set_ylabel("SPE statistic", fontsize=14)
        axs[1].set_xlabel("Observations", fontsize=14)
        axs[1].legend(loc="upper left", fontsize=12)
        axs[1].set_title("Control Chart for Phase 1 Data - SPE", loc='center', fontsize=14, fontweight=1)
        plt.show()

        return [elem >= self.ucl_t2 for elem in self.t2_scores_phase1], [elem >= self.ucl_spe for elem in self.residuals_phase1]

    def phase_two(self, data, log_scale = False):

        data = pd.DataFrame(self.scaler.transform(data))
        x = torch.Tensor(data.values)

        x_encoded, x_reconstructed, _ = self.trained_model(x)
        x_encoded = x_encoded.cpu().detach().numpy()
        x_reconstructed = pd.DataFrame(x_reconstructed.cpu().detach().numpy())
        self.t2_scores_phase2 = np.array([(sample - self.mean_vector).T @ self.inverted_cov_matrix @ (sample - self.mean_vector) for sample in x_encoded])
        self.residuals_phase2 = ((data.subtract(x_reconstructed, axis=1)) ** 2).sum(axis=1)

        if log_scale:
            self.t2_scores_phase2 = np.log(self.t2_scores_phase2)
            self.ucl_t2 = np.log(self.ucl_t2)
            self.residuals_phase2 = np.log(self.residuals_phase2)
            self.ucl_spe = np.log(self.ucl_spe)

        fig, axs = plt.subplots(2, 1, figsize=(16, 8), dpi=300)

        # T^2 chart
        axs[0].plot(self.t2_scores_phase2, marker='', color='black', linewidth=0.8, alpha=1)
        axs[0].axhline(y=self.ucl_t2, color='red', label="UCL", alpha=0.6)
        axs[0].grid(False)
        axs[0].set_yticks(color='black', fontsize=13)
        axs[0].set_xticks(color='black', fontsize=13)
        axs[0].set_ylabel("$T^2$ statistic", fontsize=14)
        axs[0].set_xlabel("Observations", fontsize=14)
        axs[0].legend(loc="upper left", fontsize=12)
        axs[0].set_title("Control Chart for Phase 1 Data - $T^2$", loc='center', fontsize=14, fontweight=1)

        # SPE chart
        axs[1].plot(self.residuals_phase2, marker='', color='black', linewidth=0.8, alpha=1)
        axs[1].axhline(y=self.ucl_spe, color='red', label="UCL", alpha=0.6)
        axs[1].grid(False)
        axs[1].set_yticks(color='black', fontsize=13)
        axs[1].set_xticks(color='black', fontsize=13)
        axs[1].set_ylabel("SPE statistic", fontsize=14)
        axs[1].set_xlabel("Observations", fontsize=14)
        axs[1].legend(loc="upper left", fontsize=12)
        axs[1].set_title("Control Chart for Phase 1 Data - SPE", loc='center', fontsize=14, fontweight=1)
        plt.show()

        return [elem >= self.ucl_t2 for elem in self.t2_scores_phase2], [elem >= self.ucl_spe for elem in self.residuals_phase2]
