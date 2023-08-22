import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from variational_autoencoder.training import TrainingModel
from variational_autoencoder.vae import VAE


file_path = '/Users/dcac/Data/TEP_clean_33vars/'
training_data = pd.read_csv(file_path+"train_fault_free_33columns.csv")
training_data = training_data[training_data["simulationRun"] == 1].iloc[:, 3:]
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(training_data))
x = torch.Tensor(data.values)

model = VAE(encoding_layers=[33, 33])
trainer = TrainingModel(learning_rate=1e-3, batch_size=10, val_size=0.20, verbose=True)
trainer.create_datasets(train_data=x)
trained_model, losses = trainer.train_model(model, patience=10, n_epochs=1000)








