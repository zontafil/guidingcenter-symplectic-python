# script used to plot charts from output
# Usage python3 plot.py -oshort outFile -olong outFile -i input_data
# olong appends useful config info in the filename
import matplotlib
from matplotlib import pyplot as plt
from config import Config
import numpy as np
import sys
import getopt
from particle import InitializationType
matplotlib.use('Agg')


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


plt.rcParams.update({'font.size': 8})

# configuration
config = Config()

# build the filenames
folderprefix = ""
longFilePrefix = ""
inputFile = config.outFile  # output of integrator is input for plotting
try:
    opts, args = getopt.getopt(sys.argv[1:], "i:", ["folderprefix=", "date="])
except getopt.GetoptError:
    print("Usage python3 plot.py -oshort outFile -olong outFile -i input_data")
    sys.exit(2)
for opt, arg in opts:
    if opt == "--folderprefix":
        folderprefix = arg
    if opt == "--date":
        longFilePrefix = arg + "_"
    if opt == "-i":
        inputFile = arg

longFilePrefix = folderprefix + longFilePrefix

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

longFilePrefix += "h{}_{}{}{}_{}_E0{}_pitch{}_{}_{}.png"\
                .format(config.h, init,
                        config.initBackwardIterations, config.initBackWardOrder,
                        config.integrator, config.E0, config.pitch,
                        config.AB_dB_Algorithm, config.emField)
shortFilePrefix = folderprefix + "last_"

x = []
y = []

data = np.genfromtxt(inputFile, delimiter=' ', skip_header=0, names=True)

dataB = None
if config.debugBfield:
    dataB = np.genfromtxt(config.debugBfieldFile, delimiter=' ', skip_header=0, names=True)

# info = "dT t/orbit: {} {}\n".format(config.h, config.stepsPerOrbit)
info = "integrator: {}\n".format(config.integrator)
info += "init bwN bwO: {} {} {}\n".format(config.initializationType,
                                          config.initBackwardIterations, config.initBackWardOrder)
info += "ABdB: {}\n".format(config.AB_dB_Algorithm)
info += "EMField: {}\n".format(config.emField)
info += "dT: {}s\n".format(config.h)
info += "T_L: {:.2e}s\n".format(data["larmorT"][0])
info += "E: {}KeV\n".format(config.E0)
info += "pitch: {}\n".format(config.pitch)
info += "m: {:.1e}kg\n".format(config.m)

fig, ax = plt.subplots(2, 2)

# energy plot
ax[0, 0].set_ylim(set_axlims(data["dE1"], 0.1))
ax[0, 0].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax[0, 0].set(xlabel="timestep", ylabel="dE/E0")
ax[0, 0].scatter(data['t'], data['dE1'], s=0.1)

# orbit
# ax[0, 1].set_ylim(set_axlims(data["y1"], 0.1))
# ax[0, 1].set_xlim(set_axlims(data["x1"], 0.1))
# ax[0, 1].set(xlabel="r", ylabel="z")
# ax[0, 1].scatter(data['x1'], data['y1'], s=0.1)
ax[0, 1].set_ylim(set_axlims(data["z1"], 0.1))
ax[0, 1].set_xlim(set_axlims(data["r1"], 0.1))
ax[0, 1].set(xlabel="r", ylabel="z")
ax[0, 1].scatter(data['r1'], data['z1'], s=0.1)

# toroidal momentum
ax[1, 0].set_ylim(set_axlims(data["p_phi"], 0.1))
ax[1, 0].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax[1, 0].set(xlabel="timestep", ylabel="p_phi err")
ax[1, 0].scatter(data['t'], data['p_phi'], s=0.1)

# x
# ax[1, 1].set_ylim(set_axlims(data["x1"], 0.1))
# ax[1, 1].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
# ax[1, 1].set(xlabel="timestep", ylabel="x1")
# ax[1, 1].scatter(data['t'], data['x1'], s=0.1)
ax[1, 1].set_ylim(set_axlims(data["u1"], 0.1))
ax[1, 1].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax[1, 1].set(xlabel="timestep", ylabel="u1")
ax[1, 1].scatter(data['t'], data['u1'], s=0.1)

fig.text(0.99, 0.99, info, va="top", ha="right")
plt.savefig(shortFilePrefix + "main.png", dpi=300)
plt.savefig(longFilePrefix, dpi=300)

plt.rcParams.update({'font.size': 14})

# u
fig, ax = plt.subplots(1, 1)
ax.set_ylim(set_axlims(data["u1"], 0.1))
ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax.set(xlabel="timestep", ylabel="u1")
ax.scatter(data['t'], data['u1'], s=0.1)

plt.savefig(shortFilePrefix + "u.png", dpi=300)

# dE
fig, ax = plt.subplots(1, 1)
ax.set_ylim(set_axlims(data["dE1"], 0.1))
ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax.set(xlabel="timestep", ylabel="dE/E0")
ax.scatter(data['t'], data['dE1'], s=4, color="black")

plt.savefig(shortFilePrefix + "dE.png", dpi=200)
# pphi
fig, ax = plt.subplots(1, 1)
ax.set_ylim(set_axlims(data["p_phi"], 0.1))
ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax.set(xlabel="timestep", ylabel="p_phi err")
ax.scatter(data['t'], data['p_phi'], s=4, color="black")
plt.savefig(shortFilePrefix + "pphi.png", dpi=200)

# orbit
fig, ax = plt.subplots(1, 1)
ax.set_ylim(set_axlims(data["z1"], 0.1))
ax.set_xlim(set_axlims(data["r1"], 0.1))
ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax.set(xlabel="r [m]", ylabel="z [m]")
ax.scatter(data['r1'], data['z1'], s=4, color="black")
plt.savefig(shortFilePrefix + "orbit.png", dpi=200)

# energy & u
if config.debugBfield:
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].set_ylim(set_axlims(dataB["muB"], 0.1))
    ax[0, 0].set_xlim(set_axlims(dataB["t"], 0.1))
    ax[0, 0].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax[0, 0].set(xlabel="t", ylabel="mu*B")
    ax[0, 0].scatter(dataB['t'], dataB['muB'], s=4, color="black")

    ax[0, 1].set_ylim(set_axlims(dataB["05u2"], 0.1))
    ax[0, 1].set_xlim(set_axlims(dataB["t"], 0.1))
    ax[0, 1].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax[0, 1].set(xlabel="t", ylabel="1/2 u**2")
    ax[0, 1].scatter(dataB['t'], dataB['05u2'], s=4, color="black")

    plt.savefig(shortFilePrefix + "Benergy.png", dpi=200)
