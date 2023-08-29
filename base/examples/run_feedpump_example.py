import pandas as pd
from matplotlib import pyplot as plt

from base import PCAModelChart, XBarChart, EWMAChart

plt.style.use("seaborn")

load_path = "../../datasets/feedwaterpump_spc.csv"

df = pd.read_csv(load_path, index_col=None)
df = df.drop(columns=["timelocal"])

df_phase1 = df.iloc[:300]
df_phase2 = df.iloc[300:].reset_index(drop=True)


# X bar chart
xchart = XBarChart(n_sample_size=3,
                   standard_deviations=3).fit(
    df_phase1=df_phase1[["temp_slipring_diff"]]
)
plt.figure(figsize=(12,8))
xchart.plot_phase1_and_2(df_phase2=df_phase2)
plt.show()


plt.savefig("present_xbarchart.png")


# EWMA chart
chart = EWMAChart(lambda_=0.3, mu_process_target=None, sigma=None).fit(df_phase2=df_phase2)
plt.figure(figsize=(12,8))
chart.plot_phase2()
plt.savefig("present_ewmachart.png")
# plt.show()


# Hotelling's T^2 with prior PCA for dimensionality reduction
chart = PCAModelChart(n_sample_size=1).fit(
    df_phase1=df_phase1,
    n_components_to_retain=None,    # leave this as None if you want to set a % variance explained
    PC_variance_explained_min=0.9,  # % variance in the data explained by the principal components
    verbose=True
)
plt.figure(figsize=(12,8))
chart.plot_phase1_and_2(df_phase2)
plt.show()

plt.savefig("present_pcachart.png")


