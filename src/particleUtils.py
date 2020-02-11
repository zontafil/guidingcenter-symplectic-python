import collections
import numpy as np


z0z1p0p1 = collections.namedtuple("z0z1p0p1", "z0 z1 p0 p1")
z1p1 = collections.namedtuple("z1p1", "z1 p1")
z2p2 = collections.namedtuple("z2p2", "z2 p2")


def cyl2cart(v, x):
    r = np.sqrt(x[0]*x[0] + x[1]*x[1])
    ret = np.zeros(3)
    ret[0] = (v[0]*x[0] - v[1]*x[1]) / r
    ret[1] = (v[0]*x[1] + v[1]*x[0]) / r
    ret[2] = v[2]
    return ret


def cart2cyl(v, x):
    r = np.sqrt(x[0]*x[0] + x[1]*x[1])
    sintheta = x[1] / r
    costheta = x[0] / r
    ret = np.zeros(3)
    ret[0] = v[0] * costheta + v[1] * sintheta
    ret[1] = - v[0] * sintheta + v[1] * costheta
    ret[2] = v[2]
    return ret


def printToFile(t, config, particle, out, timestep0=False):
    if timestep0 is True:
        z = particle.z0
        p_str = ' '.join(map(str, particle.p0))
        dE = particle.dE0
        E = particle.E0
    else:
        z = particle.z1
        p_str = ' '.join(map(str, particle.p1))
        dE = particle.dE1
        E = particle.E1
    z_str = ' '.join(map(str, z))

    r = np.sqrt(z[0]**2 + z[1]**2)
    sintheta = z[1] / r
    costheta = z[0] / r

    # compute phi component of toroidal momentum
    BdB = particle.integrator.system.fieldBuilder.compute(z)
    Adag = BdB.Adag
    Adag_phi = - Adag[0] * sintheta + Adag[1] * costheta
    p_phi = r * Adag_phi

    dz = np.linalg.norm(particle.z1 - particle.z0)
    dz = particle.dE1 - particle.dE0

    out.write("{} {} {} {} {} {} {} {}\n".format(t, t / config.stepsPerOrbit, E, z_str, r, p_str, p_phi, dz))
