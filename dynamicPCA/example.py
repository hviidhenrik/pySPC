from matplotlib import pyplot as plt

from TEP import *
from analysis import *

TEP = GetTEP()

analysis = Analysis(10)

analysis.fit(TEP)

analysis.show_plots()
plt.show()