# Frequency response for a finite cylinder
from dolfin import *
set_log_level(40) # fenics log level
from mshr import *
import numpy as np
import time

def psi(mesh, ga, lambda_min, lambda_max, N=200, dN=50, Nfactor=20):
	"""
	Frequency response of a finite cylinder. It computes the 
	average over a frequency interval of the volume average
	of the frequency response p to the Helmholz equation
	with constant Dirichlet harmonic load a part of the boundary
	which is of positive (d-1)-dimensional measure.

	The algorithm is described in arXiv: 2012.02276.

	Pre: The Fredholm alternative holds for the parameters.

	Input: Mesh mesh.
	       Boundary descriptor ga.
	       Spectral interval [lambda_min, lambda_max]
	       N, dN, Nfactor parameters for what eigenvalues
	       and eigenvalues to compute.

	Output: (\lambda_{\max}-\lambda_{\min})^{-1}\int_{\lambda_{\min}}^{\lambda_{\max}} <p_\lambda> d\lambda
	"""
	# compute eigenvalues and eigenfunctions
	dx = Measure('dx', domain=mesh)
	nu_max = 0.0
	V = FunctionSpace(mesh, "CG", 1)
	u = TrialFunction(V)
	v = TestFunction(V)
	a = dot(grad(u), grad(v))*dx
	m = u*v*dx
	bc = DirichletBC(V, Constant(0.0), ga)
	C = PETScMatrix()
	M = PETScMatrix()
	assemble(a, tensor=C)
	bc.apply(C)
	assemble(m, tensor=M)
	eigensolver = SLEPcEigenSolver(C, M)
	eigensolver.parameters["spectrum"] = "smallest real"
	while (nu_max < Nfactor*lambda_max):
		eigensolver.solve(N)
		nconv = eigensolver.get_number_converged()
		nu_max, _, _, _ = eigensolver.get_eigenpair(nconv-1)
		N = N + dN
	# compute psi
	phi_i = Function(V)
	area = assemble(Constant(1.0)*dx)
	avg_of_mean_p = 1.0
	for i in range(nconv):
		nu_i, _, phi_i.vector()[:], _ = eigensolver.get_eigenpair(i)
		phi_i_L2_squared = assemble(phi_i*phi_i*dx)
		factor_i = (area*phi_i_L2_squared)**-1
		int_phi_i = assemble(phi_i*dx)
		avg_of_mean_p = avg_of_mean_p + factor_i*int_phi_i**2*((lambda_max-lambda_min)**-1*nu_i*np.log(np.abs((nu_i-lambda_min)/(lambda_max-nu_i)))-1.0)
	return avg_of_mean_p

def Cylinder2d(x, y, mesh_density=32):
	"""
	Mesh of finite symmetric cylinder.
	
	Pre: x is ordered. y > 0.
	
	Input: Radius y at x1-coordinates x.
	
	Output: Dolfin mesh.
	
	Example: Cylinder2d([0,1], [1,1])
	"""
	upper_points = [Point(x[i], y[i]) for i in reversed(range(len(x)))]
	lower_points = [Point(x[i], -y[i]) for i in range(len(x))]
	points = lower_points + upper_points
	domain = Polygon(points)
	return generate_mesh(domain, mesh_density)

class RandomCylinder2d():
	"""
	Generator for finite cylinders.
	
	Pre: x ordered, 0 < min_radius < max_radius, mesh_density > 0.
	
	Input: Ordered x coordinates.
	       Stop at max_n if max_n > 0.
	       Minimum radius min_radius and maximal radius max_radius.
	       
	Output: Dolfin mesh with uniformly distributed random radius in
	        [min_radius, max_radius] at coordinates x.
          Mesh density mesh_density.
          Outputs radii if export_radius.
          
	Example:
		for cyl in RandomCylinder2d(x=np.linspace(0.0, 1.0, 2)):
			plot(cyl)
			plt.show()
	"""
	def __init__(self, x=[0.0, 1.0], max_n=0, min_radius=0.1, max_radius=0.5, mesh_density=32, export_radius=False):
		self.max_n = max_n
		self.n = 0
		self.min_radius = min_radius
		self.max_radius = max_radius
		self.mesh_density = mesh_density
		self.export_radius = export_radius
		self.x = x
		np.random.seed(np.uint32(time.time()*1e7%1e4))
	def __iter__(self):
		return self
	def __next__(self):
		self.n = self.n + 1
		if self.max_n > 0 and self.n > self.max_n:
			raise StopIteration
		self.radius = np.random.uniform(self.min_radius, self.max_radius, len(self.x))
		self.mesh = Cylinder2d(self.x, self.radius, self.mesh_density)
		if self.export_radius:
			return self.mesh, self.radius
		else:
			return self.mesh
