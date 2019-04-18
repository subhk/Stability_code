import sys
import os
import numpy as np
from mpi4py import MPI
from scipy.interpolate import interp1d
import scipy.io as sio



class first_derivative(object):
	"""docstring for first_derivative"""
	def __init__(self, No_Pts, Z):
		#super(first_derivative, self).__init__()
		self.Z = Z
		self.No_Pts = No_Pts

	def ddz(self):				
		if abs(np.std(np.diff(self.Z))/np.mean(np.diff(self.Z))) > 1.e-6:
			print('Values need to be equally spaced')	
			return None
		delz = self.Z[1]-self.Z[0]

		D = np.zeros( (self.No_Pts, self.No_Pts) )

		for ArrySize in range(1,self.No_Pts-1):	
			D[ArrySize,ArrySize-1] = -1.
			D[ArrySize,ArrySize+1] = +1.

		D[0,0] = -3.
		D[0,1] = 4.
		D[0,2] = -1.

		D[self.No_Pts-1,self.No_Pts-1] = 3.
		D[self.No_Pts-1,self.No_Pts-2] = -4.
		D[self.No_Pts-1,self.No_Pts-3] = 1.

		
		D = np.divide(D, 2.*delz)

		return D



class Trapezoidal(object):
	"""docstring for Trapezoidal"""
	def __init__(self, No_Pts, Z, j, k, Uvel, Uzz, rho_z):
		#super(Trapezoidal, self).__init__()
		self.No_Pts = No_Pts
		self.Z = Z
		self.j = j
		self.k = k
		self.Uvel = Uvel
		self.Uzz = Uzz
		self.rho_z = rho_z

	def integ_func(self):
		Hmax = 2.*np.max(abs(self.Z))

		integ_randm = np.zeros( (self.No_Pts,3) )	

		tmp1 = np.sin(self.j*np.pi*(self.Z+0.5*Hmax)/Hmax)
		tmp2 = np.sin(self.k*np.pi*(self.Z+0.5*Hmax)/Hmax)
		tmp = tmp1*tmp2 

		Uzz1 = np.reshape(self.Uzz,(1, -1))
		rhoz = np.reshape(self.rho_z,(1, -1))

		#print(np.shape(Uzz1))

		integ_randm[:,0] = (2./Hmax)*self.Uvel*tmp #np.multiply(self.Uvel,tmp_)
		integ_randm[:,1] = (2./Hmax)*Uzz1*tmp #np.multiply(self.Uzz ,tmp_)
		integ_randm[:,2] = (2./Hmax)*rhoz*tmp #np.multiply(self.rho_z,tmp_) 

		Sum = np.zeros(3)

		#print(np.shape(integ_randm))

		delz = self.Z[1]-self.Z[0]
		for ArrySize in range(3):
			Sum[ArrySize] = np.trapz(integ_randm[0:self.No_Pts,ArrySize],self.Z) #, delz, 1)

		return Sum






