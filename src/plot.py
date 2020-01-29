# script used to plot charts from output
# Usage python3 plot.py -oshort outFile -olong outFile -i input_data
# olong appends useful config info in the filename

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
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
inputFile = config.outFile  # output of integrator is input for plotting
try:
    opts, args = getopt.getopt(sys.argv[1:], "i:", ["olong=", "oshort="])
except getopt.GetoptError:
    print("Usage python3 plot.py -oshort outFile -olong outFile -i input_data")
    sys.exit(2)
for opt, arg in opts:
    if opt == "--oshort":
        outFileShort = arg
    if opt == "--olong":
        outFileLong = arg
    if opt == "-i":
        inputFile = arg

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

outFileLong += "h{}_{}{}{}_{}_mu{}_{}_{}.png"\
                .format(config.h, init,
                        config.initBackwardIterations, config.initBackWardOrder,
                        config.integrator, config.mu,
                        config.AB_dB_Algorithm, config.emField)

x = []
y = []

data = np.genfromtxt(inputFile, delimiter=' ', skip_header=0, names=True)

info = "timestep t/orbit: {} {}\n".format(config.h, config.stepsPerOrbit)
info += "integrator: {}\n".format(config.integrator)
info += "init bwN bwO: {} {} {}\n".format(config.initializationType,
                                          config.initBackwardIterations, config.initBackWardOrder)
info += "ABdB: {}\n".format(config.AB_dB_Algorithm)
info += "EMField: {}\n".format(config.emField)
info += "mu: {}\n".format(config.mu)

fig, ax = plt.subplots(2, 2)

# energy plot
ax[0, 0].set_ylim(set_axlims(data["dE1"], 0.1))
ax[0, 0].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax[0, 0].set(xlabel="timestep", ylabel="dE/E0")
ax[0, 0].scatter(data['t'], data['dE1'], s=0.1)

# orbit
ax[0, 1].set(xlabel="r", ylabel="z")
ax[0, 1].scatter(data['r1'], data['z1'], s=0.1)

# toroidal momentum
ax[1, 0].set_ylim(set_axlims(data["Adag_phi"], 0.1))
ax[1, 0].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax[1, 0].set(xlabel="timestep", ylabel="A_dag_phi")
ax[1, 0].scatter(data['t'], data['Adag_phi'], s=0.1)

fig.text(0.99, 0.99, info, va="top", ha="right")
plt.savefig(outFileShort, dpi=300)
plt.savefig(outFileLong, dpi=300)
