# script used to plot charts from output
# Usage python3 plot.py -oshort outFile -olong outFile
# olong appends useful config info in the filename

import matplotlib.pyplot as plt
from config import Config
import numpy as np
import sys
import getopt
from particle import InitializationType


def set_axlims(series, marginfactor):
    """
    Fix for a scaling issue with matplotlibs scatterplot and small values.
    Takes in a pandas series, and a marginfactor (float).
    A marginfactor of 0.2 would for example set a 20% border distance on both sides.
    Output:[bottom,top]
    To be used with .set_ylim(bottom,top)
    """
    minv = series.min()
    maxv = series.max()
    datarange = maxv-minv
    border = abs(datarange*marginfactor)
    maxlim = maxv+border
    minlim = minv-border

    return minlim, maxlim


plt.rcParams.update({'font.size': 5})

# configuration
config = Config()

# build the filenames
outFileShort = ""
outFileLong = ""
try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["olong=", "oshort="])
except getopt.GetoptError:
    print("Usage python3 plot.py -oshort outFile -olong outFile")
    sys.exit(2)
for opt, arg in opts:
    if opt == "--oshort":
        outFileShort = arg
    if opt == "--olong":
        outFileLong = arg

if config.initializationType == InitializationType.HAMILTONIAN:
    init = "ham"
elif config.initializationType == InitializationType.LAGRANGIAN:
    init = "lag"
elif config.initializationType == InitializationType.MANUAL_Z0Z1:
    init = "man"
elif config.initializationType == InitializationType.MANUAL:
    init = "man"
else:
    init = config.initializationType

outFileShort += "main.png"
outFileLong += "h{}_{}{}{}_{}_mu{}_{}_{}.png"\
                .format(config.h, init,
                        config.initBackwardIterations, config.initBackWardOrder,
                        config.integrator, config.mu,
                        config.AB_dB_Algorithm, config.emField)

x = []
y = []

data = np.genfromtxt(config.outFile, delimiter=' ', skip_header=0, names=True)

info = "timestep t/orbit: {} {}\n".format(config.h, config.stepsPerOrbit)
info += "integrator: {}\n".format(config.integrator)
info += "init bwN bwO: {} {} {}\n".format(config.initializationType,
                                          config.initBackwardIterations, config.initBackWardOrder)
info += "ABdB: {}\n".format(config.AB_dB_Algorithm)
info += "EMField: {}\n".format(config.emField)
info += "mu: {}\n".format(config.mu)

fig, (ax, ax2) = plt.subplots(1, 2)
ax.set_ylim(set_axlims(data["dE1"], 0.1))
ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax.set(xlabel="timestep", ylabel="dE/E0")
ax.scatter(data['t'], data['dE1'], s=1)
fig.text(0.99, 0.99, info, va="top", ha="right")

ax2.set(xlabel="r", ylabel="z")
ax2.scatter(data['r1'], data['z1'], s=1)
plt.savefig(outFileShort, dpi=300)
plt.savefig(outFileLong, dpi=300)
