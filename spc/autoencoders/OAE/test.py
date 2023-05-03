import pandas as pd
from control_charts import ProcessMonitoring

file_path = '/Users/dcac/Data/TEP_clean_33vars'
training_data = pd.read_csv(file_path+"train_fault_free_33columns.csv")
training_data = training_data[training_data["simulationRun"] == 1].iloc[:, 3:]
testing_data = pd.read_csv(file_path+"train_faulty_33columns.csv")
testing_data = testing_data[testing_data["simulationRun"] == 1].iloc[:, 3:]
