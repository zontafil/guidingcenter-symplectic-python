"""
A CLASS TO HANDLE EQDSK FILE
- read data (instantiate)
- plot data (draw_...)
- evaluate magnetic field and derivatives (B_and_dB)
https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
"""

import numpy as np
import re
from scipy.interpolate import RectBivariateSpline as spl2d
from scipy.interpolate import UnivariateSpline as spl1d


def read_data(fp):
    lines = []
    while True:
        line = fp.readline()
        if not line:
            break
        pattern = r'[+-]?\d*[\.]?\d+(?:[Ee][+-]?\d+)?'
        data = re.findall(pattern, line)
        lines.append(data)
    return [item for sublist in lines for item in sublist]


class EqdskReader():

    def __init__(self, fname):

        # open the file for reading
        fp = open(fname, "r")

        # read (neqdsk,2000) (case(i),i=1,6),idum,nw,nh
        header = fp.readline().split()
        self.nr = int(header[-2])  # number of R points for psi(R,z)
        self.nz = int(header[-1])  # number of z points for psi(R,z)

        data = read_data(fp)
        i = 0
        # read (neqdsk,2020) rdim,zdim,rcentr,rleft,zmid
        self.rdim = float(data[i])
        i = i+1
        self.zdim = float(data[i])
        i = i+1
        self.rcentr = float(data[i])
        i = i+1
        self.rleft = float(data[i])
        i = i+1
        self.zmid = float(data[i])
        i = i+1

        # read (neqdsk,2020) rmaxis,zmaxis,simag,sibry,bcentr
        self.rmaxis = float(data[i])
        i = i+1
        self.zmaxis = float(data[i])
        i = i+1
        self.simag = float(data[i])
        i = i+1
        self.sibry = float(data[i])
        i = i+1
        self.bcentr = float(data[i])
        i = i+1

        # read (neqdsk,2020) current,simag,xdum,rmaxis,xdum
        # read (neqdsk,2020) zmaxis,xdum,sibry,xdum,xdum
        self.current = float(data[i])
        i = i+10
        # read (neqdsk,2020) (fpol(i),i=1,nw)
        self.fpol = np.array(
            [float(num) for num in data[i:i+self.nr]])
        i = i+self.nr
        # read (neqdsk,2020) (pres(i),i=1,nw)
        self.pres = np.array(
            [float(num) for num in data[i:i+self.nr]])
        i = i+self.nr
        # read (neqdsk,2020) (ffprim(i),i=1,nw)
        self.ffprim = np.array(
            [float(num) for num in data[i:i+self.nr]])
        i = i+self.nr
        # read (neqdsk,2020) (pprime(i),i=1,nw)
        self.pprime = np.array(
            [float(num) for num in data[i:i+self.nr]])
        i = i+self.nr
        # read (neqdsk,2020) ((psirz(i,j),i=1,nw),j=1,nh)
        self.psirz = np.array(
            [float(num) for num in data[i:i+self.nr*self.nz]]).reshape(
                self.nz, self.nr).transpose()
        i = i+self.nr*self.nz
        # read (neqdsk,2020) (qpsi(i),i=1,nw)
        self.qpsi = np.array(
            [float(num) for num in data[i:i+self.nr]])
        i = i+self.nr
        # read (neqdsk,2022) nbbbs,limitr
        self.nbbbs = int(data[i])
        i = i+1
        self.limitr = int(data[i])
        i = i+1
        # read (neqdsk,2020) (rbbbs(i),zbbbs(i),i=1,nbbbs)
        self.rbbbs = np.array([float(num)
                               for num in data[i:i+2*self.nbbbs-1:2]])
        self.zbbbs = np.array(
            [float(num) for num in data[
                i+1:i+1+2*self.nbbbs:2]])
        i = i+2*self.nbbbs
        # read (neqdsk,2020) (rlim(i),zlim(i),i=1,limitr)
        self.rlim = np.array([float(num)
                              for num in data[i:i+2*self.limitr-1:2]])
        self.zlim = np.array(
            [float(num) for num in data[
                i+1:i+1+2*self.limitr:2]])
        i = i+2*self.nbbbs

        # Set up the
        # minimum and maximum in z for the separatrix curve
        # for the quick check to evaluate the fpol function
        self.sepminz = min(self.zbbbs)
        self.sepmaxz = max(self.zbbbs)

        fp.close()

        # Set up splines
        Rs = np.linspace(self.rleft, self.rleft+self.rdim, self.nr)
        Zs = np.linspace(self.zmid-self.zdim/2.0,
                         self.zmid+self.zdim/2.0, self.nz)
        self.psi_spl = spl2d(Rs, Zs, self.psirz, kx=4, ky=4)
        psis = np.linspace(self.simag, self.sibry, self.nr)
        self.fpol_spl = spl1d(psis, self.fpol, k=4)

        self.r_min = self.rleft
        self.r_max = self.r_min + self.rdim
        self.r_grid = (self.r_max - self.r_min) / (self.nr - 1)
        self.z_min = self.zmid - self.zdim / 2.0
        self.z_max = self.zmid + self.zdim / 2.0
        self.z_grid = (self.z_max - self.z_min) / (self.nz - 1)
