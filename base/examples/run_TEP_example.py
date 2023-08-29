from matplotlib import pyplot as plt

from base import PCAModelChart
from dynamicPCA.TEP import GetTEP

plt.style.use("seaborn")

df = GetTEP()

df_phase1 = df.iloc[:600]
df_phase2 = df.iloc[600:]

chart = PCAModelChart(n_sample_size=1).fit(
    df_phase1=df_phase1,
    n_components_to_retain=None,    # leave this as None if you want to set a % variance explained
    PC_variance_explained_min=0.6,  # % variance in the data explained by the principal components
    verbose=True
)

chart.plot_phase1_and_2(df_phase2)
plt.show()
