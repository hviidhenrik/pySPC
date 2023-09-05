import pandas as pd
from autoencoders.oae.control_charts import ProcessMonitoring

training_data = pd.read_csv("datasets/tep_training.csv")
testing_data = pd.read_csv("datasets/tep_testing.csv")

SPC = ProcessMonitoring(alpha=0.05)
SPC.fit(data=training_data, encoding_layers=[33, 10], orthogonality_regularization=1, norm="l2",
        nr_epochs=1000, patience=10, batch_size=10, verbose=True)
SPC.plot_correlation_matrix()
y_train_pred = SPC.plot_phase1(log_scale=True)
y_test_pred = SPC.plot_phase2(data=testing_data, log_scale=True)
out_of_control_obs = testing_data.iloc[161, :]
SPC.t2_contribution_plots(ooc_observation=out_of_control_obs, method="monte_carlo", samples=1000)
SPC.spe_contribution_plots(out_of_control_obs)
