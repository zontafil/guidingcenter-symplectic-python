from config import Config
from particle import Particle


def printToFile(t, config, particle, out):
    z1 = ' '.join(map(str, particle.z1))
    out.write("{} {} {} {}\n".format(t, particle.dE1, z1, particle.r1(), particle.p1))


# configuration
config = Config()

# open output file
out = open(config.outFile, "w+")

# create a particle
particle = Particle(config)
particle.initialize()

print("time step: ", config.h)
print("Initialization: ")
print("z_init: " + str(particle.z0))
print("z0: " + str(particle.z1))

out.write("t dE1 x1 y1 z1 u1 r1 px1 py1 pz1 pu1\n")
printToFile(0, config, particle, out)

#  ******
# MAIN LOOP
#  ******
for t in range(1, config.nsteps):
    # PRINT TO SCREEN EVERY N STEPS
    if (t % config.printTimestepMult) == 0:
        print("Timestep " + str(t))

    particle.stepForward(t)

    # PRINT TO FILE
    if (t % config.fileTimestepMult) == 0:
        printToFile(t, config, particle, out)

    # EXIT IF THE ERROR IS TOO HIGH
    if config.exitOnError:
        if abs(particle.dE1) > config.errorThreshold:
            print("Timestep: " + str(t) + "\tError too high! Exiting.")
            break
