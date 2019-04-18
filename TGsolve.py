"""
Stability analysis of Taylor Goldstein equation

"""

import sys
import os
from time import clock
import numpy as np
from mpi4py import MPI
import scipy.io as sio
from multiprocessing import Pool

from Matrix import MatrixSolver

def vel_profile(No_Pts, Z):
	Uvel = np.zeros(No_Pts)
	Uvel = np.tanh(Z)

	return Uvel

def den_profile(No_Pts, Z):
	rho_ = np.zeros(No_Pts)
	Ratio = 3.0
	eps = 0.0
	rho_ = -np.tanh(Ratio*(Z+eps)) 

	return rho_

def LinearSolver(No_Pts, Z, invRe, invPr, imode):
	
	J = 0.1
	alpha = 0.1

	Uvel = vel_profile(No_Pts, Z)
	rho_ = den_profile(No_Pts, Z)

	start = clock()	
	Obj_ = MatrixSolver(No_Pts, Z, Uvel, J, rho_, alpha, invRe, invPr, imode)
	Ri, gamma_max = Obj_.mSolver()

	print(gamma_max)
	elapsed = (clock() - start)
	print('Total time elapsed in the Solver: ', elapsed)

if __name__ == '__main__':
	# Initialisation the parameters
	No_Pts = 401
	Hmax = 10.
	Z = np.linspace(-0.5*Hmax, 0.5*Hmax, No_Pts)
	
	invRe = 1.e-4
	invPr = 1.

	imode = 1
	LinearSolver(No_Pts, Z, invRe, invPr, imode)


