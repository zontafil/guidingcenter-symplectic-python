from config import Config
from particle import Particle
from particleUtils import printToFile
import sys
import getopt
from integrators.integratorFactory import integratorFactory
from particleUtils import z0z1p0p1
import numpy as np



# configuration
config = Config()

# parse command line
outFile = config.outFile
try:
    opts, args = getopt.getopt(sys.argv[1:], "o:")
except getopt.GetoptError:
    print("Usage python3 main.py -o outFile")
    sys.exit(2)
for opt, arg in opts:
    if opt == "-o":
        outFile = arg

# create a particle
particle = Particle(config, config.z0, config.p0, config.z1, config.p1)
particle.initialize(config.initializationType)
for i in range(config.initBackwardIterations):
    particle.backwardInitializationIteration(config.initBackWardOrder)

print("Saving to {}".format(outFile))
print("Time step: ", config.h)
print("Initialization: ")
print("z_init: " + str(particle.z0))
print("z0: " + str(particle.z1))
print("E0: " + str(particle.E0))

# open output file
out = open(outFile, "w+")
out.write("t norbit dE1 x1 y1 z1 u1 r1 px1 py1 pz1 pu1 p_phi dz\n")
printToFile(0, config, particle, out, timestep0=True)
printToFile(1, config, particle, out)
out.close()
out = open(outFile, "a+")

#  ******
# MAIN LOOP
#  ******
reinit = True
for t in range(2, config.nsteps):
    # PRINT TO SCREEN EVERY N STEPS
    if (t % config.printTimestepMult) == 0:
        print("Timestep " + str(t))

    particle.stepForward(t)

    if reinit is False:
        if (particle.z0[3] / particle.z1[3]) < 0:
            print("reinit")
            auxiliaryIntegrator = integratorFactory(config.auxiliaryIntegrator, config)
            while True:
                # print(np.abs(particle.z0[3]))
                if np.abs(particle.z0[3]) < 1E-6:
                    break
                points = z0z1p0p1(z1=particle.z0, z0=None, p0=None, p1=None)
                points = auxiliaryIntegrator.stepForward(points, config.h / 10000)
                particle.z0 = points.z2
            print("reinit done")
            reinit = True
            particle.initialize(config.initializationType)
            for i in range(config.initBackwardIterations):
                particle.backwardInitializationIteration(config.initBackWardOrder)
            print(particle.z0)
            print(particle.z1)

    # PRINT TO FILE
    if (t % config.fileTimestepMult) == 0:
        printToFile(t, config, particle, out)

    # EXIT IF THE ERROR IS TOO HIGH
    if config.exitOnError:
        if abs(particle.dE1) > config.errorThreshold:
            print("Timestep: " + str(t) + "\tError too high! Exiting.")
            break
