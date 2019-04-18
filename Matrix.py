import sys
import os
from time import clock
import numpy as np
from numpy import linalg as SOLVER

from utility import first_derivative, Trapezoidal

# Kronecker Delta function
def Kron_Delta(x,y):
	if x==y:
		return 1.
	else:
		return 0.	

class MatrixSolver(object):
	"""docstring for MatrixSolver"""
	def __init__(self, No_Pts, Z, Uvel, J, rho_, alpha, invRe, invPr, imode):
		#super(MatrixSolver, self).__init__()
		self.No_Pts = No_Pts
		self.Z = Z
		self.Uvel = Uvel
		self.J = J
		self.rho_ = rho_
		self.alpha = alpha
		self.invRe = invRe
		self.invPr = invPr
		self.imode = imode

	def mSolver(self):
		Hmax = 2.*np.max(abs(self.Z))

		tmp1 = np.zeros( (self.No_Pts,self.No_Pts) )
		tmp2 = np.zeros( (self.No_Pts,self.No_Pts) )
		tmp3 = np.zeros( (self.No_Pts,self.No_Pts) )		

		Matrix = np.zeros( (2*self.No_Pts,2*self.No_Pts), dtype=complex )

		#phi_prime = np.zeros( (self.No_pts, self.imode) )
		#W_prime   = np.zeros( (self.No_pts, self.imode) )
		#rho_prime = np.zeros( (self.No_pts, self.imode) )

		obj_0 = first_derivative(self.No_Pts, self.Z)
		D1 = obj_0.ddz()

		U1 = np.reshape(self.Uvel,(-1, 1))
		r1 = np.reshape(self.rho_,(-1, 1))

		Uvel_z  = D1.dot(U1) #D1*U1
		Uvel_zz = D1.dot(Uvel_z)

		#print(np.shape(Uvel_z))

		rho_z = D1.dot(r1)

		## calculating the gradient Richardson number
		Ri = np.zeros(self.No_Pts)
		Ri = -self.J*rho_z/Uvel_z**2.

		for iArry in range(self.No_Pts):
			for jArry in range(self.No_Pts):
				
				obj_1 = Trapezoidal(self.No_Pts, self.Z, iArry, jArry, self.Uvel, Uvel_zz, rho_z)
				Sum = obj_1.integ_func()	
				
				tmp1[iArry, jArry] = Sum[0]
				tmp2[iArry, jArry] = Sum[1]

				if self.J != 0:
					tmp3[iArry, jArry] = Sum[2] 

				Qi = (iArry*np.pi/Hmax)**2. + self.alpha**2.
				Qj = (jArry*np.pi/Hmax)**2. + self.alpha**2.

				Matrix[iArry, jArry] = \
				(1./Qi)*(Qj*tmp1[iArry,jArry]+tmp2[iArry,jArry]) - 1j*self.invRe*Qi*Kron_Delta(iArry,jArry)/self.alpha

				Matrix[iArry, jArry+self.No_Pts] = \
				self.J*Kron_Delta(iArry,jArry)/Qi

				Matrix[iArry+self.No_Pts, jArry] = \
				-tmp3[iArry, jArry]

				Matrix[iArry+self.No_Pts, jArry+self.No_Pts] = \
				tmp1[iArry, jArry] - 1j*self.invRe*self.invPr*Qi*Kron_Delta(iArry,jArry)/self.alpha

		start = clock()		
		eig_val, eig_func = SOLVER.eig(Matrix)
		elapsed = (clock() - start)
		print('Time elapsed in EigenSolver: ', elapsed)

		#eig_val = np.diag(eig_val)
		#print(eig_val)				
		ct_real = eig_val.real
		ct_imag = eig_val.imag

		indx  = np.argsort(ct_imag)[::-1]

		c_imag = np.zeros( len(indx) )
		c_real = np.zeros( len(indx) )

		#print(indx[0])

		for iArry in range( len(indx)-1 ):
			c_imag[iArry] = ct_imag[indx[iArry]]
			c_real[iArry] = ct_real[indx[iArry]]

			
		if self.imode == 1:	
			gamma_max = self.alpha*c_imag[0]
		else:
			gamma_max = self.alpha*c_imag[0:self.imode-1]

		return Ri, gamma_max	
